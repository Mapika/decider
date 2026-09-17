"""Intent routing with described options and confidence-gated actions (the TypeSafe "confidence-gated routing" pattern).
   python examples/routing_with_confidence.py [model]

Known weakness (v5 and v6): do not add a catch-all "other" option next to a generic one such as "support"; a complaint is then
sent to "other" with high confidence.  Scope checks written as free-form yes/no questions are also unreliable at this model size."""
import sys
sys.path.insert(0, ".")
from decider.infer import Decider

d = Decider(sys.argv[1] if len(sys.argv) > 1 else "Mapika/decider-2b")
QUESTIONS = {
    "action": {"type": "choice", "instructions": "What is the user trying to do?",
               "criteria": {"check_balance": "View the account balance or recent transactions",
                            "approve_transfer": {"what": "Approve or confirm a pending outgoing transfer", "not_for": "Asking about a transfer's status"},
                            "support": "Get help with a problem, a complaint, or a question about the product"}},
    "urgent": {"type": "noul", "instructions": "Does the message convey urgency?"},
    "frustration": {"type": "score", "instructions": "How frustrated is the user?", "criteria": ["calm", "irritated", "angry or threatening to leave"]},
}
for msg in ["how much do i have left in checking?", "yes go ahead and send the 2,400 to Marta", "third time I'm asking. the app logs me out every time. fix it today or I'm gone",
            "did my transfer to Marta go through or not"]:
    a = d.system_one(msg, QUESTIONS)["answers"]; act = a["action"]
    if act["confidence"] < 0.5:
        route = "-> human (model unsure)"
    elif act["choice"] == "approve_transfer":
        route = "-> execute" if act["confidence"] > 0.9 else "-> ask the user to confirm"        # destructive action: higher bar
    else:
        route = f"-> {act['choice']}"
    print(f"{msg[:70]:72s} {act['choice']:17s} p={act['confidence']:.2f}  urgent={a['urgent']['noul']:.2f}  frustration={a['frustration']['score']:.2f}  {route}")
