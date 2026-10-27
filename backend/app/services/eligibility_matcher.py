import re
def match_eligibility(requirements,profile):
    criteria=[x.strip(" .-\n") for x in re.split(r"[;\n]|(?<=[.!?])\s+",requirements or "") if len(x.strip())>12]
    satisfied=[]; unmet=[]; unknown=[]
    for item in criteria[:20]:
        low=item.lower(); reason=""
        if any(k in low for k in ("engineering","computer science","b.tech","degree")):
            words=[w for w in (str(profile.get("branch",""))+" "+str(profile.get("degree",""))).lower().split() if len(w)>3]
            if words and any(w in low for w in words): reason="Profile field matches stated discipline"
            else: unknown.append({"criterion":item,"reason":"Discipline details need confirmation"}); continue
        elif any(k in low for k in ("year","final year","first year","second year","third year")):
            year=profile.get("year_of_study")
            if year is None: unknown.append({"criterion":item,"reason":"Study year is missing"}); continue
            if str(year) in low or (year==4 and "final" in low): reason=f"Year of study: {year}"
            else: unmet.append({"criterion":item,"reason":f"Profile year is {year}"}); continue
        elif "india" in low or "indian" in low:
            if "india" in str(profile.get("location","India")).lower(): reason="Profile location includes India"
            else: unmet.append({"criterion":item,"reason":"Location does not match"}); continue
        else: unknown.append({"criterion":item,"reason":"Requires manual confirmation"}); continue
        satisfied.append({"criterion":item,"reason":reason})
    status="not_enough_information" if unknown else ("likely_match" if not unmet else "criteria_not_met")
    return {"status":status,"satisfied":satisfied,"not_satisfied":unmet,"needs_confirmation":unknown,"note":"Matching uses available profile fields; verify requirements with the publisher."}

