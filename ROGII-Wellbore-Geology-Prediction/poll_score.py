import subprocess, csv, io, time, sys
for i in range(90):
    r = subprocess.run(["python", "-m", "kaggle", "competitions", "submissions",
                        "-c", "rogii-wellbore-geology-prediction", "--csv"],
                       capture_output=True, text=True)
    x = list(csv.DictReader(io.StringIO(r.stdout)))[0]
    ps = x.get("publicScore", "")
    st = x.get("status", "").split(".")[-1]
    if ps and ps[0].isdigit():
        print(f"SCORED after {i} polls: status={st} publicScore={ps}", flush=True)
        break
    time.sleep(60)
else:
    print("still pending after long poll", flush=True)
