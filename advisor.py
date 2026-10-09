"""
advisor.py
==========
Rule-based water quality advisor.

For each parameter that falls outside WHO/BIS approximate guideline limits,
the advisor generates practical, plain-language remediation steps.

If a GEMINI_API_KEY environment variable is set, it also attempts to
generate a short AI-powered explanation via the Gemini API.  If the API
call fails for any reason, it falls back to the rule-based text gracefully.
"""

import os
from config import PARAM_CONFIG, DISCLAIMER

# ---------------------------------------------------------------------------
# Rule-based remediation database
# ---------------------------------------------------------------------------
REMEDIATION_RULES = {
    "ph": {
        "too_high": (
            "High pH (Alkaline Water)\n"
            "- Add a small amount of food-grade citric acid or use an acid-neutralising filter.\n"
            "- Reverse Osmosis (RO) filtration effectively lowers high pH.\n"
            "- Avoid using alkaline water for infant formula.\n"
            "- Report persistently high pH to your local water authority."
        ),
        "too_low": (
            "Low pH (Acidic Water)\n"
            "- Install a calcite or soda-ash neutralising filter.\n"
            "- Boiling can slightly raise pH for mildly acidic water.\n"
            "- Acidic water may corrode pipes — flush taps for 30 seconds before use.\n"
            "- Report consistently low pH to your local water utility."
        ),
    },
    "Hardness": {
        "too_high": (
            "High Hardness (Hard Water)\n"
            "- Install a water softener (ion-exchange resin) to reduce calcium and magnesium.\n"
            "- Reverse Osmosis (RO) filtration reduces hardness significantly.\n"
            "- Descale kettles and appliances regularly.\n"
            "- Hard water is not a direct health hazard but may affect taste and appliances."
        ),
        "too_low": None,  # Low hardness is generally not a concern
    },
    "Solids": {
        "too_high": (
            "High Total Dissolved Solids (TDS)\n"
            "- Reverse Osmosis (RO) is the most effective treatment for high TDS.\n"
            "- Distillation also removes dissolved solids.\n"
            "- Avoid using high-TDS water for cooking and drinking without treatment.\n"
            "- Identify and remove the source of contamination (industrial discharge, saltwater intrusion)."
        ),
        "too_low": None,
    },
    "Chloramines": {
        "too_high": (
            "High Chloramines\n"
            "- Use an activated carbon (charcoal) filter — it removes chloramines effectively.\n"
            "- Unlike chlorine, boiling does NOT remove chloramines.\n"
            "- Catalytic carbon filters are especially effective.\n"
            "- Avoid using high-chloramine water for fish tanks or kidney dialysis without treatment.\n"
            "- Report to your water supplier if levels are consistently above guidelines."
        ),
        "too_low": None,
    },
    "Sulfate": {
        "too_high": (
            "High Sulfate\n"
            "- Reverse Osmosis (RO) effectively removes sulfate.\n"
            "- Anion exchange filters also reduce sulfate levels.\n"
            "- High sulfate can cause a laxative effect — avoid drinking until treated.\n"
            "- Test for the source: natural geology or industrial pollution."
        ),
        "too_low": None,
    },
    "Conductivity": {
        "too_high": (
            "High Conductivity\n"
            "- High conductivity indicates excess dissolved ions — treat with RO or distillation.\n"
            "- Identify the source: runoff from fertilisers, road salt, or industrial discharge.\n"
            "- Report abnormal conductivity spikes to the local environment agency."
        ),
        "too_low": None,
    },
    "Organic_carbon": {
        "too_high": (
            "High Organic Carbon\n"
            "- Activated carbon (GAC) filters are highly effective at removing organic compounds.\n"
            "- Boiling water reduces some organic contaminants but does not eliminate all.\n"
            "- High organic carbon can react with disinfectants to form harmful by-products.\n"
            "- Ensure the water source is protected from agricultural or sewage runoff."
        ),
        "too_low": None,
    },
    "Trihalomethanes": {
        "too_high": (
            "High Trihalomethanes (THMs)\n"
            "- THMs form when chlorine reacts with organic matter — reduce both sources.\n"
            "- Activated carbon (GAC) filters effectively remove THMs.\n"
            "- Allow treated tap water to sit uncovered in a jug for 1–2 hours (some THMs volatilise).\n"
            "- Avoid hot showers with high-THM water as inhalation is a concern.\n"
            "- Long-term high exposure is linked to health risks — prioritise treatment."
        ),
        "too_low": None,
    },
    "Turbidity": {
        "too_high": (
            "High Turbidity (Cloudy Water)\n"
            "- Let turbid water settle in a clean container for several hours, then carefully pour off the clearer top layer.\n"
            "- Filter through a clean cloth, coffee filter, or sand filter to remove suspended particles.\n"
            "- Boil water after filtration to kill any pathogens that may be attached to particles.\n"
            "- For reliable treatment, use a multi-stage filter (sediment → carbon → UV).\n"
            "- High turbidity is a warning sign — do not drink without treatment."
        ),
        "too_low": None,
    },
}


def check_parameters(input_values: dict) -> list[dict]:
    """
    Compare input values against WHO/BIS guideline limits.

    Parameters
    ----------
    input_values : dict
        Dictionary of {feature_name: value}.

    Returns
    -------
    list[dict]
        Each dict has keys: param, display_name, value, unit, status,
        who_min, who_max, advice.
        status ∈ {"OK", "Too High", "Too Low"}
    """
    results = []
    for param, value in input_values.items():
        cfg = PARAM_CONFIG.get(param, {})
        who_min = cfg.get("who_min")
        who_max = cfg.get("who_max")
        unit    = cfg.get("unit", "")
        display = cfg.get("display_name", param)

        status = "OK"
        advice = None

        if who_max is not None and value > who_max:
            status = "Too High"
            rule   = REMEDIATION_RULES.get(param, {})
            advice = rule.get("too_high", f"Value is above the recommended limit of {who_max} {unit}.")
        elif who_min is not None and value < who_min:
            status = "Too Low"
            rule   = REMEDIATION_RULES.get(param, {})
            advice = rule.get("too_low", f"Value is below the recommended limit of {who_min} {unit}.")

        results.append(
            {
                "param":        param,
                "display_name": display,
                "value":        value,
                "unit":         unit,
                "status":       status,
                "who_min":      who_min,
                "who_max":      who_max,
                "advice":       advice,
            }
        )
    return results


def get_ai_advice(input_values: dict, prediction: int, probability: float) -> str | None:
    """
    Try to generate AI-powered advice using the Gemini API.

    Returns None (falls back to rule-based) if GEMINI_API_KEY is not set
    or if the API call fails.

    Parameters
    ----------
    input_values : dict
        The water parameter values entered by the user.
    prediction : int
        0 = Unsafe, 1 = Safe.
    probability : float
        Model confidence (0–1).

    Returns
    -------
    str | None
        Markdown-formatted advice string, or None on failure.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None

    try:
        import google.generativeai as genai  # type: ignore

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")

        status_text = "SAFE" if prediction == 1 else "UNSAFE"
        param_lines = "\n".join(
            f"  - {PARAM_CONFIG[k]['display_name']}: {v} {PARAM_CONFIG[k]['unit']}"
            for k, v in input_values.items()
        )

        prompt = (
            f"You are a water quality expert helping a non-technical community member "
            f"understand their water test results. Be friendly, clear, and concise.\n\n"
            f"The AI model predicted the water is {status_text} "
            f"(confidence: {probability:.1%}).\n\n"
            f"Water parameters tested:\n{param_lines}\n\n"
            f"In 3–5 sentences, explain:\n"
            f"1. What the overall result means for their health.\n"
            f"2. Which parameter(s) are most concerning (if any) and why.\n"
            f"3. The single most important action they should take right now.\n\n"
            f"Keep the language simple — as if speaking to someone with no science background.\n"
            f"End with: '{DISCLAIMER}'"
        )

        response = model.generate_content(prompt)
        return response.text

    except Exception as exc:  # noqa: BLE001
        # Silently fall back to rule-based advice
        print(f"[advisor] Gemini API call failed: {exc}")
        return None


def build_rule_based_summary(check_results: list[dict]) -> str:
    """
    Compile all out-of-range advisories into a single Markdown string.

    Parameters
    ----------
    check_results : list[dict]
        Output of check_parameters().

    Returns
    -------
    str
        Markdown-formatted summary of issues and advice.
    """
    issues = [r for r in check_results if r["status"] != "OK"]
    if not issues:
        return (
            "All parameters are within approximate guideline limits.\n\n"
            "Continue regular testing and maintain good water storage hygiene.\n\n"
            f"{DISCLAIMER}"
        )

    lines = [f"{len(issues)} parameter(s) require attention:\n"]
    for r in issues:
        lines.append(f"{r['display_name']} — {r['status']}")
        lines.append(r["advice"] or "Consult a water quality professional.")
        lines.append("")

    lines.append(f"---\n{DISCLAIMER}")
    return "\n".join(lines)
