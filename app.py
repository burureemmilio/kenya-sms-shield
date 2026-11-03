
"""KenyaSMS Shield: sequential binary + scam-severity SMS assessment.
Run from project root: streamlit run app.py
"""

import streamlit as st
from src.advanced import assess_message, BUNDLE_PATH
import joblib
from src.advanced import AdvancedFeatures
st.set_page_config(
    page_title="KenyaSMS Shield",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ KenyaSMS Shield")
st.caption(
    "Kenya-focused, explainable SMS scam research prototype · "
    "Binary detection followed by scam-severity classification"
)

bundle = joblib.load(BUNDLE_PATH)

EXAMPLES = {
    "Choose an example": "",
    "Personal Kiswahili text": "Niaje bro, nimefika home salama. Tutaonana kesho.",
    "Illustrative M-Pesa receipt": (
        "TQ82AB91CD Confirmed. Ksh2,500.00 sent to JOHN DOE "
        "0712345678 on 2/11/26 at 6:20 PM. New M-Pesa balance "
        "is Ksh1,200.00. Transaction cost, Ksh25.00. Safaricom."
    ),
    "Illustrative fake prize": (
        "CONGRATULATIONS! Umeshinda Ksh 250,000. Claim now. "
        "Pay Ksh 500 registration fee to receive your prize: bit.ly/claim-ke"
    ),
    "Illustrative reversal request": (
        "Habari, nimetuma pesa kwa namba yako kimakosa. "
        "Tafadhali nitumie PIN yako au tuma Ksh 2,000 "
        "kwa hii number ili nireverse."
    ),
    "Illustrative fake bank alert": (
        "KCB ALERT: Your account is BLOCKED. Verify your PIN "
        "and OTP immediately at http://kcb-secure-verify.xyz"
    ),
}

left, right = st.columns([1, 1])

with left:
    choice = st.selectbox("Try a sample message", list(EXAMPLES))

    message = st.text_area(
        "Paste SMS message",
        value=EXAMPLES[choice],
        height=190,
        placeholder="Paste the complete SMS here..."
    )

    analyse = st.button(
        "Analyse SMS",
        type="primary",
        use_container_width=True
    )

with right:
    st.subheader("How the system works")

    st.markdown("""
1. **Binary gate:** estimates whether the message resembles scam or legitimate SMS.
2. **Risk policy:** low scores are SAFE; borderline scores are SUSPICIOUS.
3. **Severity model:** messages crossing the scam threshold are classified as moderate-risk or high-risk scams.
4. **Explanation:** shows influential words and Kenya-specific indicators.
""")

    st.info(
        "This tool analyses message text only. It cannot verify the sender, "
        "transaction, URL ownership, or whether a receipt is genuine."
    )


if analyse:
    if not message.strip():
        st.warning("Enter a message first.")

    else:
        result = assess_message(message, bundle)
        tier = result["risk_tier"]

        if tier == 0:
            st.success("SAFE — no strong scam signal detected")
        elif tier == 1:
            st.warning("SUSPICIOUS — verify independently before acting")
        else:
            st.error(
                "HIGH-RISK SCAM — do not send money, share credentials, or open links"
            )

        a, b, c = st.columns(3)

        a.metric("Risk score (0–100)", f"{result['risk_score']:.1f}")
        b.metric("Binary scam score", f"{result['binary_score']:.3f}")
        c.metric(
            "Second-stage class",
            result["severity_prediction"] or "Not triggered"
        )

        st.caption(result["disclaimer"])

        # Rule-based warning indicators
        st.subheader("Rule-based warning indicators")

        if result["indicators"]:
            for item in result["indicators"]:
                st.markdown(f"- {item}")
        else:
            st.write("No configured warning indicators found.")

        # Interpretation and safety guidance
        with st.expander("Interpretation and safety guidance", expanded=True):
            st.markdown("""
- A SAFE result is **not a guarantee**. Scams can be short, new, or deliberately written like ordinary messages.
- A structured M-Pesa-looking receipt is not proof of payment. Check the actual balance/transaction in the official app or statement.
- Never share an M-Pesa PIN, bank PIN, password, or OTP.
- Do not use a phone number or link supplied by a suspicious message to verify it. Contact the institution using a trusted official channel.
- The model is trained on a small, annotated dataset. Its scores are not calibrated probabilities and the output must not automatically block messages or initiate financial actions.
""")
