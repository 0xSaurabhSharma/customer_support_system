# streamlit.py
import streamlit as st
import requests
from datetime import datetime

st.set_page_config(page_title="Flipkart Product Assistant", layout="wide")

API_URL = st.sidebar.text_input("API base URL", value="http://localhost:8000")
st.sidebar.markdown("---")
st.sidebar.write("This Streamlit app talks to the FastAPI backend `/get` endpoint.")

if "messages" not in st.session_state:
    st.session_state.messages = []  # list of (who, text, ts)
if "pending_user_msg" not in st.session_state:
    st.session_state.pending_user_msg = None  # message to send next run

def send_to_backend(user_msg: str):
    """Send message to backend and append assistant reply."""
    try:
        with st.spinner("Thinking..."):
            resp = requests.post(f"{API_URL}/get", data={"msg": user_msg}, timeout=60)
            if resp.status_code == 200:
                assistant = resp.text
            else:
                assistant = f"[error {resp.status_code}] {resp.text}"
    except Exception as e:
        assistant = f"[network error] {e}"

    st.session_state.messages.append(("Assistant", assistant, datetime.now().strftime("%H:%M")))
    st.rerun()

# Layout
col1, col2 = st.columns([3, 2])

with col1:
    st.markdown("<h2 style='margin:0 0 10px 0'>Flipkart Product Assistant</h2>", unsafe_allow_html=True)
    chat_box = st.container()

    # display messages
    for who, text, ts in st.session_state.messages:
        if who == "You":
            st.markdown(
                f"<div style='display:flex;justify-content:flex-end;margin:6px 0'>"
                f"<div style='background:#facc15;color:#111;padding:10px 14px;border-radius:12px;max-width:70%;'>"
                f"<b>You</b><div style='font-size:14px;margin-top:4px'>{text}</div>"
                f"<div style='text-align:right;font-size:10px;opacity:0.7'>{ts}</div></div></div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div style='display:flex;justify-content:flex-start;margin:6px 0'>"
                f"<div style='background:rgba(255,255,255,0.06);color:#fff;padding:10px 14px;border-radius:12px;max-width:70%;'>"
                f"<b>Assistant</b><div style='font-size:14px;margin-top:4px'>{text}</div>"
                f"<div style='text-align:right;font-size:10px;opacity:0.7'>{ts}</div></div></div>",
                unsafe_allow_html=True
            )

    # message input
    with st.form("message_form", clear_on_submit=True):
        user_input = st.text_input("Type a message...", key="input")
        submitted = st.form_submit_button("Send")
        if submitted and user_input:
            # Step 1: show user msg immediately
            ts = datetime.now().strftime("%H:%M")
            st.session_state.messages.append(("You", user_input, ts))
            st.session_state.pending_user_msg = user_input
            st.rerun()

# Step 2: If we have a pending msg, send it now
if st.session_state.pending_user_msg:
    msg_to_send = st.session_state.pending_user_msg
    st.session_state.pending_user_msg = None
    send_to_backend(msg_to_send)

with col2:
    st.markdown("### Controls")
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.session_state.pending_user_msg = None
        st.rerun()
    st.markdown("---")
    st.markdown("**Sample queries**")
    if st.button("Low budget headphone"):
        ts = datetime.now().strftime("%H:%M")
        st.session_state.messages.append(("You", "Can you tell me the low budget headphone?", ts))
        st.session_state.pending_user_msg = "Can you tell me the low budget headphone?"
        st.rerun()
    if st.button("Top rated phone under 20k"):
        ts = datetime.now().strftime("%H:%M")
        st.session_state.messages.append(("You", "Top rated phone under 20,000 INR?", ts))
        st.session_state.pending_user_msg = "Top rated phone under 20,000 INR?"
        st.rerun()
