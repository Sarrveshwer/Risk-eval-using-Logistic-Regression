# Machine Failure Predictor

## Why I actually built this

The goal is simple: **saving lives and cash.**

In a real factory, when a machine snaps, it's not just a big repair bill—it's dangerous. People get hurt when things fail without warning. I wanted to see if I could use raw sensor data (like temperature, RPM, and torque) to catch a breakdown *before* it happens, giving workers time to hit the kill switch or call for maintenance.

---

## 🏗️ Better Tech (MySQL > CSV)

Loads of people just toss a CSV into a script and call it a day. I didn't want to do that. I moved the whole backend to **MySQL**. 
*   **Stability**: It handles thousands of rows without breaking a sweat.
*   **Reality**: Most factories use databases, not Excel files.
*   **Testing**: I built a custom scenario generator (`create_test_tables.py`) that splits data into specific failure modes like Heat Failure or Tool Wear. This lets me prove the model actually knows *why* the machine is struggling.

---

## 🧠 The "Hurdle Model" Novelty

I realized pretty early that even the best specialist can miss things. I built the **Hurdle Model Architecture** as a fallback system to make sure nothing slips through the cracks.

The data has to clear two separate hurdles:

### **Hurdle 1: The Safety Net (Binary Scorer)**
This is the broad brain. Its job is to calculate a **Global Risk Score**. Its main purpose? **Catching random errors.** If the machine is acting weird but doesn't fit a "standard" failure type, this hurdle catches it anyway. 

### **Hurdle 2: The Specialist (Multinomial Diagnosis)**
If (and only if) the first hurdle is cleared, the **Diagnostic Specialist** wakes up. It tries to pinpoint exactly what's wrong—is it Heat (**HDF**), Tool Wear (**TWF**), or Overstrain? If the specialist is confused, we still know there's a problem because it already cleared the first hurdle.

---

## 📈 Catching the Trend (Feature Engineering)

Machine failure isn't a single "moment"; it’s a process. If you only look at one second of data, you miss the story. I built features that look at the **history**, not just the "now":

1.  **Rolling Mean**: Smoothing out the jittery sensor noise.
2.  **Volatility (Std Dev)**: This is the big one. If the sensor starts shaking wildly, it’s a massive red flag.
3.  **Delta & Rolling Delta**: Looking at the *change* in data. Is the temperature rising faster than it was a minute ago? That’s how we catch failures early.

---

## 📊 The "Needle in a Haystack" Problem

Here's a technical challenge: **Accuracy is a lie.** 

In my data, 97% of the runs were perfectly normal. If a model just guesses "Everything is fine" all day, it gets a 97% accuracy score—but it misses every single life-threatening failure. I ignored accuracy and focused on **ROC-AUC** and **Precision-Recall** to make sure we find the rare 3% that actually matter.

---

## 🎞️ Project Showcase

<details>
  <summary><b>Click to see the System UI & Dashboards</b></summary>
  
  ### 1. Live Monitoring Dashboard
  <img src="images/frontend.png" alt="Live Dashboard"/>
  
  ### 2. Real-time Log Engine
  <img src="images/logs.png" alt="Log Engine"/>
  
  ### 3. Deep Performance Metrics
  <img src="images/info2.png" alt="Performance Metrics"/>
  
  ### 4. Global Risk Analysis (Tier 1)
  <img src="images/Dashboard_ai4i2020.png" alt="Tier 1 Analysis"/>
  
  ### 5. Diagnostic Specialization (Tier 2)
  <img src="images/Dashboard_Classifications_ai4i2020.png" alt="Tier 2 Specialization"/>
</details>

---

## 🛠️ How to run it

1.  Clone the repo.
2.  Set up your MySQL DB (credentials in `ml_engine.py`).
3.  Run `python manage.py runserver`.
4.  Go to the **Test** tab and trigger a "Heat Failure" preset to watch the AI catch it live.

**Goal**: Keep the machines running and the people safe.
