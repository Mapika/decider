"""Quick abstention sanity battery: does the model still pick real answers when 'none of the above' is offered?"""
import sys
from .infer import Decider
cases = [("My card was charged twice for the same purchase.", "Which department should handle this?", ["billing", "technical support", "sales"], "billing"),
         ("The app crashes every time I open the settings page on Android.", "Which department should handle this?", ["billing", "technical support", "sales"], "technical support"),
         ("Can I get a quote for 50 seats on the enterprise plan?", "Which department should handle this?", ["billing", "technical support", "sales"], "sales"),
         ("Please cancel my subscription and refund last month.", "What does the customer want?", ["a refund", "a feature", "technical help"], "a refund"),
         ("Turn off the kitchen lights.", "What is the intent?", ["smart home control", "set alarm", "play music"], "smart home control"),
         ("The quarterly report shows revenue fell 12%.", "What is the financial sentiment?", ["bearish", "neutral", "bullish"], "bearish"),
         ("Where is your office located?", "Which department should handle this?", ["billing", "technical support", "sales"], "none of the above"),
         ("What is the capital of France?", "Which department should handle this?", ["billing", "technical support", "sales"], "none of the above")]
for name in sys.argv[1:]:
    d = Decider(name, use_graphs=False); ok = 0
    for ctx, q, opts, gold in cases:
        r = d.decide(ctx, [{"question": q, "options": opts + ["none of the above"]}])[0]; ok += r["choice"] == gold
        print(f"  {name.split('/')[-2] if name.endswith('model') else name:12s} gold={gold:18s} -> {r['choice']:18s} {r['confidence']:.2f}")
    print(f"== {name}: {ok}/{len(cases)} correct with 'none of the above' offered")
