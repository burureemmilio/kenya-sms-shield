import unittest
from src.advanced import AdvancedFeatures, KENYA_PATTERNS, BUNDLE_PATH, assess_message
import joblib

class AdvancedFeatureTests(unittest.TestCase):
    def test_feature_extractor_has_kenyan_signals(self):
        text="TQ82AB91CD Confirmed. Ksh 500 sent. M-Pesa balance Ksh 200. Transaction cost Ksh 15."
        names=list(AdvancedFeatures().get_feature_names_out())
        vals=AdvancedFeatures().transform([text])[0]
        features=dict(zip(names,vals))
        self.assertEqual(features["ke_mpesa_terms"],1)
        self.assertEqual(features["ke_mpesa_receipt_structure"],1)
    def test_model_bundle_smoke(self):
        self.assertTrue(BUNDLE_PATH.exists(),"Train model with python -m src.advanced first")
        bundle=joblib.load(BUNDLE_PATH)
        result=assess_message("Your KCB account is blocked. Send your OTP immediately.",bundle)
        self.assertIn(result["risk_label"],{"SAFE","SUSPICIOUS","HIGH-RISK SCAM"})
        self.assertGreaterEqual(result["risk_score"],0)
        self.assertLessEqual(result["risk_score"],100)

if __name__=="__main__": unittest.main()
