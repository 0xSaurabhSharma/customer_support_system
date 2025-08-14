# app.py
import streamlit as st
import requests
import pandas as pd
from typing import Dict

# ----- Config -----
st.set_page_config(page_title="RAG Chatbots", layout="wide")

BASE_URL = "http://localhost:8000"
RETRIEVAL_RAG_CHAT_URL = f"{BASE_URL}/get"
AGENTIC_RAG_CHAT_URL = f"{BASE_URL}/chat"
UPLOAD_DOC_URL = f"{BASE_URL}/upload-doc"
LIST_DOCS_URL = f"{BASE_URL}/list-docs"
DELETE_DOC_URL = f"{BASE_URL}/delete-doc"

# ----- Helper: model options (display -> value sent to API) -----
MODEL_OPTIONS: Dict[str, str] = {
    "Gemini (flash-lite)": "gemini-2.0-flash-lite",
    "gpt-4.1": "gpt-4.1",
    "gpt-4.1-mini": "gpt-4.1-mini",
}

# ----- Ensure session state keys exist -----
if "messages_retrieval" not in st.session_state:
    st.session_state.messages_retrieval = []
if "messages_agentic" not in st.session_state:
    st.session_state.messages_agentic = []
# session_id will always be string (never None) to avoid Pydantic null errors
if "session_id" not in st.session_state:
    st.session_state.session_id = ""

# ----- Layout: Sidebar controls -----
st.sidebar.title("RAG Controls")
mode = st.sidebar.radio("Mode", ("Retrieval RAG", "Agentic RAG"))

st.sidebar.markdown("---")
st.sidebar.header("Agent Settings")
selected_model_display = st.sidebar.selectbox("Select model", list(MODEL_OPTIONS.keys()))
selected_model = MODEL_OPTIONS[selected_model_display]

st.sidebar.markdown("---")
st.sidebar.header("Document Management")
uploaded_file = st.sidebar.file_uploader("Upload a document", type=["pdf", "docx", "html"])

# Upload button (only enabled if a file selected)
if uploaded_file is not None:
    if st.sidebar.button("Upload & Index"):
        try:
            # Build files tuple so requests sends correct content-type and filename
            file_bytes = uploaded_file.read()
            files = {"file": (uploaded_file.name, file_bytes, uploaded_file.type or "application/octet-stream")}
            res = requests.post(UPLOAD_DOC_URL, files=files, timeout=60)
            try:
                res.raise_for_status()
                st.sidebar.success(f"Uploaded and indexed: {uploaded_file.name}")
            except requests.HTTPError:
                st.sidebar.error(f"Upload failed ({res.status_code}): {res.text}")
        except requests.exceptions.RequestException as e:
            st.sidebar.error(f"Error uploading file: {e}")

st.sidebar.markdown("---")
if st.sidebar.button("List Indexed Documents"):
    try:
        res = requests.get(LIST_DOCS_URL, timeout=30)
        res.raise_for_status()
        docs = res.json()
        if docs:
            df = pd.DataFrame(docs)
            st.sidebar.write("Indexed documents:")
            st.sidebar.dataframe(df)
            # allow user to select & delete
            try:
                ids = [int(d.get("id")) for d in docs]
                sel = st.sidebar.selectbox("Select document id to delete", options=ids)
                if st.sidebar.button("Delete selected document"):
                    try:
                        del_res = requests.post(DELETE_DOC_URL, json={"file_id": sel}, timeout=30)
                        if del_res.status_code == 200:
                            st.sidebar.success(f"Deleted doc {sel}")
                        else:
                            st.sidebar.error(f"Delete failed ({del_res.status_code}): {del_res.text}")
                    except requests.exceptions.RequestException as e:
                        st.sidebar.error(f"Error while deleting: {e}")
            except Exception:
                # If docs don't have numeric ids or other shape, ignore delete UI
                pass
        else:
            st.sidebar.info("No documents indexed yet.")
    except requests.exceptions.RequestException as e:
        st.sidebar.error(f"Error listing documents: {e}")

st.sidebar.markdown("---")
if st.sidebar.button("Clear all session chat"):
    st.session_state.messages_retrieval = []
    st.session_state.messages_agentic = []
    st.session_state.session_id = ""
    st.sidebar.success("Cleared chat and session id.")


# ----- Page Title -----
st.title(f"{mode} Chat")

# ----- Retrieval RAG -----
if mode == "Retrieval RAG":
    col1, col2 = st.columns([3, 1])
    with col1:
        # Display chat history
        for message in st.session_state.messages_retrieval:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Input
        prompt = st.chat_input("Ask something (retrieval)...")
        if prompt:
            st.session_state.messages_retrieval.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        # use form data to match existing endpoint
                        res = requests.post(RETRIEVAL_RAG_CHAT_URL, data={"msg": prompt}, timeout=60)
                        # show debug info for development
                        st.write("Status:", res.status_code)
                        try:
                            st.json(res.json())
                        except Exception:
                            st.text(res.text)

                        res.raise_for_status()
                        answer = res.text
                        st.markdown(answer)
                        st.session_state.messages_retrieval.append({"role": "assistant", "content": answer})
                    except requests.exceptions.RequestException as e:
                        st.error(f"Error communicating with API: {e}")
                        st.session_state.messages_retrieval.append({"role": "assistant", "content": "Error: Could not connect to the RAG service."})

    with col2:
        st.markdown("### Quick controls")
        st.write("Model (preview):", selected_model_display)
        st.write("Session ID:", st.session_state.session_id or "(not set)")
        st.markdown(
            "Use the sidebar to upload/index documents or to list/delete indexed docs. "
            "If you see a `422` from the API, check the session id and model strings."
        )

# ----- Agentic RAG -----
else:
    col1, col2 = st.columns([3, 1])
    with col1:
        # Display chat history
        for message in st.session_state.messages_agentic:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        prompt = st.chat_input("Ask something (agentic)...")
        if prompt:
            st.session_state.messages_agentic.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Prepare payload; ensure session_id is a string (never null)
            payload = {
                "question": prompt,
                "session_id": st.session_state.session_id or "",
                "model": selected_model,
            }

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        res = requests.post(AGENTIC_RAG_CHAT_URL, json=payload, headers={"Content-Type": "application/json"}, timeout=120)
                        st.write("Status:", res.status_code)

                        # try parse json for debugging
                        res_json = None
                        try:
                            res_json = res.json()
                            st.json(res_json)
                        except ValueError:
                            st.text(res.text)

                        if res.status_code == 200:
                            # prefer typical key "answer"; fallback to others
                            result = res_json or {}
                            answer = result.get("answer") or result.get("Answer") or result.get("result") or result.get("response") or res.text
                            # update session id if backend returned one
                            returned_sid = (result.get("session_id") if isinstance(result, dict) else None) or ""
                            if returned_sid:
                                st.session_state.session_id = returned_sid
                            st.markdown(answer)
                            st.session_state.messages_agentic.append({"role": "assistant", "content": answer})
                        else:
                            # show helpful error
                            err_text = res.text
                            st.error(f"API Error {res.status_code}: {err_text}")
                            st.session_state.messages_agentic.append({"role": "assistant", "content": f"Error: API {res.status_code}"})

                    except requests.exceptions.RequestException as e:
                        st.error(f"Error communicating with API: {e}")
                        st.session_state.messages_agentic.append({"role": "assistant", "content": "Error: Could not connect to the RAG service."})

    with col2:
        st.markdown("### Session & Model")
        st.write("Model:", selected_model_display)
        st.write("Session ID:", st.session_state.session_id or "(not set)")
        if st.button("Reset session id"):
            st.session_state.session_id = ""
            st.success("Session id reset.")
        st.markdown("---")
        st.markdown("Debugging tips:")
        st.markdown(
            "- If you get `422 Unprocessable Entity`, it usually means a field type mismatch (e.g., `session_id` was `null`).\n"
            "- This UI ensures `session_id` is never `null` — it sends `\"\"` if session is empty.\n"
            "- Check the OpenAPI docs at `http://localhost:8000/docs` for the exact schema expected by the server."
        )

















# import streamlit as st
# import requests
# import json
# import io
# import pandas as pd

# # Set up page configuration
# st.set_page_config(
#     page_title="RAG Chatbots",
#     layout="wide",
# )

# # FastAPI endpoint URLs
# BASE_URL = "http://localhost:8000"
# RETRIEVAL_RAG_CHAT_URL = f"{BASE_URL}/get"
# AGENTIC_RAG_CHAT_URL = f"{BASE_URL}/chat"
# UPLOAD_DOC_URL = f"{BASE_URL}/upload-doc"
# LIST_DOCS_URL = f"{BASE_URL}/list-docs"
# DELETE_DOC_URL = f"{BASE_URL}/delete-doc"

# # --- Sidebar for Mode Selection ---
# st.sidebar.title("Choose a RAG Mode")
# mode = st.sidebar.radio(
#     "Select an option:",
#     ("Retrieval RAG", "Agentic RAG")
# )

# st.title(f"{mode} Chat")

# # --- Retrieval RAG Chat UI ---
# if mode == "Retrieval RAG":
#     if "messages_retrieval" not in st.session_state:
#         st.session_state.messages_retrieval = []

#     # Display chat messages from history
#     for message in st.session_state.messages_retrieval:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])

#     # Handle user input
#     if prompt := st.chat_input("What is up?"):
#         # Add user message to chat history
#         st.session_state.messages_retrieval.append({"role": "user", "content": prompt})
#         with st.chat_message("user"):
#             st.markdown(prompt)

#         # Send request to FastAPI
#         with st.chat_message("assistant"):
#             with st.spinner("Thinking..."):
#                 try:
#                     response = requests.post(RETRIEVAL_RAG_CHAT_URL, data={"msg": prompt})
#                     response.raise_for_status()
#                     answer = response.text
#                     st.markdown(answer)
#                     st.session_state.messages_retrieval.append({"role": "assistant", "content": answer})
#                 except requests.exceptions.RequestException as e:
#                     st.error(f"Error communicating with API: {e}")
#                     st.session_state.messages_retrieval.append({"role": "assistant", "content": "Error: Could not connect to the RAG service."})

# # --- Agentic RAG Chat UI ---
# elif mode == "Agentic RAG":
    
#     # Initialize session state for agentic mode
#     if "messages_agentic" not in st.session_state:
#         st.session_state.messages_agentic = []
#     if "session_id" not in st.session_state:
#         st.session_state.session_id = None
    
#     # Sidebar for document management
#     st.sidebar.markdown("---")
#     st.sidebar.header("Document Management")
    
#     # Document Upload
#     uploaded_file = st.sidebar.file_uploader("Upload a document", type=['pdf', 'docx', 'html'])
#     if uploaded_file and st.sidebar.button("Upload & Index"):
#         try:
#             files = {'file': uploaded_file.getvalue()}
#             response = requests.post(UPLOAD_DOC_URL, files=files)
#             response.raise_for_status()
#             st.sidebar.success(f"File {uploaded_file.name} uploaded successfully!")
#         except requests.exceptions.RequestException as e:
#             st.sidebar.error(f"Error uploading file: {e}")

#     # Document Listing and Deletion
#     st.sidebar.markdown("---")
#     if st.sidebar.button("List Indexed Documents"):
#         try:
#             response = requests.get(LIST_DOCS_URL)
#             response.raise_for_status()
#             docs = response.json()
#             if docs:
#                 df = pd.DataFrame(docs)
#                 st.sidebar.write("Currently Indexed Documents:")
#                 st.sidebar.dataframe(df)
#             else:
#                 st.sidebar.info("No documents have been indexed yet.")
#         except requests.exceptions.RequestException as e:
#             st.sidebar.error(f"Error listing documents: {e}")

#     # Display chat messages
#     for message in st.session_state.messages_agentic:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])

#     # Handle user input for agentic chat
#     if prompt := st.chat_input("What is up?"):
#         # Add user message to chat history
#         st.session_state.messages_agentic.append({"role": "user", "content": prompt})
#         with st.chat_message("user"):
#             st.markdown(prompt)

#         # Prepare and send request to FastAPI
#         with st.chat_message("assistant"):
#             with st.spinner("Thinking..."):
#                 try:
#                     payload = {
#                         "question": prompt,
#                         "session_id": st.session_state.session_id,
#                         "model": "gemini-2.0-flash-lite" 
#                     }
#                     response = requests.post(
#                         AGENTIC_RAG_CHAT_URL, 
#                         json=payload,
#                         headers={"Content-Type": "application/json"}
#                     )
#                     st.write("Status:", response.status_code)
#                     try:
#                         st.json(response.json())
#                     except Exception:
#                         st.text(response.text)
#                     response.raise_for_status()
#                     result = response.json()
#                     answer = result.get("answer", "No answer found.")
#                     st.session_state.session_id = result.get("session_id")
#                     st.markdown(answer)
#                     st.session_state.messages_agentic.append({"role": "assistant", "content": answer})
#                 except requests.exceptions.RequestException as e:
#                     st.error(f"Error communicating with API: {e}")
#                     st.session_state.messages_agentic.append({"role": "assistant", "content": "Error: Could not connect to the RAG service."})