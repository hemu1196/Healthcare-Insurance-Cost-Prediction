import pandas as pd 
import numpy as np
import os

def generate_dataset(num_rows=100000, seed=42):
    np.random.seed(seed)
    
    print(f"Generating synthetic medical cost dataset with {num_rows} rows...")
    
    # 1. Demographics
    age = np.random.randint(18, 85, size=num_rows)
    gender = np.random.choice(["Male", "Female"], size=num_rows, p=[0.49, 0.51])
    income = np.random.exponential(scale=50000, size=num_rows) + 20000
    income = np.clip(income, 15000, 250000)
    children = np.random.choice([0, 1, 2, 3, 4, 5], size=num_rows, p=[0.45, 0.25, 0.18, 0.08, 0.03, 0.01])
    region = np.random.choice(["Northeast", "Southeast", "Southwest", "Northwest"], size=num_rows)
    employment_status = np.random.choice(["Employed", "Self-Employed", "Retired", "Unemployed"], 
                                         size=num_rows, p=[0.60, 0.15, 0.15, 0.10])
    marital_status = np.random.choice(["Single", "Married", "Divorced", "Widowed"], 
                                       size=num_rows, p=[0.35, 0.45, 0.15, 0.05])
    
    # 2. Lifestyle
    # BMI distribution, partially correlated with Age and Income
    bmi_base = np.random.normal(26, 5, size=num_rows)
    bmi_age_factor = (age - 18) * 0.05
    bmi_income_factor = - (income - 50000) / 100000 * 0.5
    bmi = bmi_base + bmi_age_factor + bmi_income_factor
    bmi = np.clip(bmi, 15.0, 50.0)
    
    smoker = np.random.choice(["Yes", "No"], size=num_rows, p=[0.18, 0.82])
    alcohol = np.random.choice(["No", "Low", "Medium", "High"], size=num_rows, p=[0.30, 0.40, 0.22, 0.08])
    exercise_frequency = np.random.choice(["Rarely", "1-2 times/week", "3-4 times/week", "Daily"], 
                                           size=num_rows, p=[0.35, 0.30, 0.20, 0.15])
    diet_quality = np.random.choice(["Poor", "Average", "Good", "Excellent"], 
                                     size=num_rows, p=[0.20, 0.45, 0.25, 0.10])
    sleep_hours = np.clip(np.random.normal(7.0, 1.2, size=num_rows), 4.0, 10.0)
    stress_level = np.random.randint(1, 11, size=num_rows)
    water_intake = np.clip(np.random.normal(2.2, 0.7, size=num_rows), 1.0, 4.0)
    active_hours = np.clip(np.random.normal(1.5, 0.9, size=num_rows), 0.0, 6.0)
    smokeless_tobacco = np.random.choice(["Yes", "No"], size=num_rows, p=[0.05, 0.95])
    
    # 3. Clinical / Medical History
    # Chronic conditions probability model based on age & BMI
    def prob_disease(age_val, bmi_val, base_prob, age_coef=0.03, bmi_coef=0.04):
        z = base_prob + age_coef * (age_val - 40) + bmi_coef * (bmi_val - 25)
        prob = 1 / (1 + np.exp(-z))
        return np.random.binomial(1, prob)
        
    diabetes = np.array([prob_disease(a, b, -2.5, 0.04, 0.08) for a, b in zip(age, bmi)])
    hypertension = np.array([prob_disease(a, b, -2.0, 0.05, 0.07) for a, b in zip(age, bmi)])
    heart_disease = np.array([prob_disease(a, b, -3.5, 0.06, 0.04) for a, b in zip(age, bmi)])
    asthma = np.random.choice([0, 1], size=num_rows, p=[0.90, 0.10])
    kidney_disease = np.array([prob_disease(a, b, -4.0, 0.05, 0.03) for a, b in zip(age, bmi)])
    copd = np.array([prob_disease(a, b, -4.5, 0.07, 0.02) for a, b in zip(age, bmi)])
    # Smoker increases COPD probability significantly
    copd = np.where((smoker == "Yes") & (np.random.rand(num_rows) < 0.25), 1, copd)
    
    joint_pain = np.where(age > 50, np.random.choice([0, 1], size=num_rows, p=[0.40, 0.60]), 
                          np.random.choice([0, 1], size=num_rows, p=[0.85, 0.15]))
    allergies = np.random.choice([0, 1], size=num_rows, p=[0.75, 0.25])
    anxiety_depression = np.random.choice([0, 1], size=num_rows, p=[0.80, 0.20])
    thyroid_issue = np.random.choice([0, 1], size=num_rows, p=[0.92, 0.08])
    
    systolic_bp = 100 + (age * 0.3) + (bmi * 0.5) + (hypertension * 20) + np.random.normal(0, 5, size=num_rows)
    systolic_bp = np.clip(systolic_bp, 90, 180).astype(int)
    
    diastolic_bp = 60 + (age * 0.15) + (bmi * 0.3) + (hypertension * 12) + np.random.normal(0, 3, size=num_rows)
    diastolic_bp = np.clip(diastolic_bp, 60, 110).astype(int)
    
    cholesterol = 150 + (age * 0.8) + (bmi * 1.2) + (heart_disease * 30) + np.random.normal(0, 15, size=num_rows)
    cholesterol = np.clip(cholesterol, 120, 300).astype(int)
    
    blood_sugar = 80 + (diabetes * 60) + (bmi * 0.8) + np.random.normal(0, 10, size=num_rows)
    blood_sugar = np.clip(blood_sugar, 70, 200).astype(int)
    
    hba1c = 4.0 + (diabetes * 2.5) + (bmi * 0.04) + np.random.normal(0, 0.3, size=num_rows)
    hba1c = np.clip(hba1c, 4.0, 10.0)
    
    family_history_diabetes = np.random.choice([0, 1], size=num_rows, p=[0.70, 0.30])
    family_history_hypertension = np.random.choice([0, 1], size=num_rows, p=[0.65, 0.35])
    family_history_heart_disease = np.random.choice([0, 1], size=num_rows, p=[0.80, 0.20])
    
    # 4. Healthcare Utilization
    hospital_visits = np.random.poisson(lam=0.2 + 0.5 * diabetes + 0.4 * heart_disease + 0.3 * kidney_disease, size=num_rows)
    hospital_visits = np.clip(hospital_visits, 0, 15)
    
    emergency_visits = np.random.poisson(lam=0.1 + 0.3 * heart_disease + 0.2 * copd, size=num_rows)
    emergency_visits = np.clip(emergency_visits, 0, 8)
    
    specialist_visits = np.random.poisson(lam=0.5 + 0.8 * diabetes + 1.2 * heart_disease + 0.6 * thyroid_issue, size=num_rows)
    specialist_visits = np.clip(specialist_visits, 0, 12)
    
    telehealth_visits = np.random.poisson(lam=0.8 + 0.4 * anxiety_depression, size=num_rows)
    telehealth_visits = np.clip(telehealth_visits, 0, 20)
    
    preventative_visits = np.random.choice([0, 1, 2, 3], size=num_rows, p=[0.30, 0.50, 0.15, 0.05])
    
    prescriptions_count = np.random.poisson(lam=0.5 + 1.5 * diabetes + 1.2 * hypertension + 2.0 * heart_disease + 0.8 * anxiety_depression, size=num_rows)
    prescriptions_count = np.clip(prescriptions_count, 0, 15)
    
    prior_authorizations = np.random.poisson(lam=0.1 * specialist_visits + 0.2 * hospital_visits, size=num_rows)
    prior_authorizations = np.clip(prior_authorizations, 0, 10)
    
    denials_count = np.random.poisson(lam=0.05 * prior_authorizations, size=num_rows)
    denials_count = np.clip(denials_count, 0, 5)
    
    adherence_score = np.random.beta(a=8, b=2, size=num_rows) # mostly high adherence
    satisfaction_score = np.random.choice([1, 2, 3, 4, 5], size=num_rows, p=[0.05, 0.10, 0.20, 0.40, 0.25])
    vaccination_status = np.random.choice(["Fully Vaccinated", "Partially Vaccinated", "Not Vaccinated"], 
                                           size=num_rows, p=[0.75, 0.15, 0.10])
    
    # 5. Insurance Policy Details
    insurance_plan = np.random.choice(["Basic", "Standard", "Premium"], size=num_rows, p=[0.40, 0.45, 0.15])
    
    # Policy values depending on plan
    out_of_pocket_max = np.where(insurance_plan == "Basic", np.random.normal(7000, 500), 
                                 np.where(insurance_plan == "Standard", np.random.normal(4500, 400), np.random.normal(2000, 300)))
    out_of_pocket_max = np.clip(out_of_pocket_max, 1000, 8500)
    
    deductible = np.where(insurance_plan == "Basic", np.random.normal(4000, 300), 
                          np.where(insurance_plan == "Standard", np.random.normal(2000, 250), np.random.normal(500, 100)))
    deductible = np.clip(deductible, 250, 5000)
    
    copay = np.where(insurance_plan == "Basic", np.random.normal(40, 5), 
                     np.where(insurance_plan == "Standard", np.random.normal(25, 3), np.random.normal(15, 2)))
    copay = np.clip(copay, 10, 50)
    
    urban_rural = np.random.choice(["Urban", "Suburban", "Rural"], size=num_rows, p=[0.40, 0.45, 0.15])
    work_life_balance = np.random.choice([1, 2, 3, 4, 5], size=num_rows, p=[0.10, 0.20, 0.40, 0.20, 0.10])
    
    # 6. Targets and intermediate variables
    # Medical costs calculation (Regression Target)
    base_cost = 2500.0
    age_cost = (age ** 1.3) * 15.0
    bmi_cost = np.where(bmi > 30, (bmi - 30) * 350.0 + 3000.0, 0.0)
    smoker_cost = np.where(smoker == "Yes", 15000.0, 0.0)
    
    # Interaction effect
    smoker_bmi_interaction = np.where((smoker == "Yes") & (bmi > 30), 12000.0, 0.0)
    
    chronic_cost = (diabetes * 4000.0 + 
                    hypertension * 2500.0 + 
                    heart_disease * 10000.0 + 
                    kidney_disease * 8000.0 + 
                    copd * 5000.0 + 
                    asthma * 1500.0 +
                    anxiety_depression * 2000.0)
    
    utilization_cost = (hospital_visits * 2500.0 + 
                        emergency_visits * 1800.0 + 
                        specialist_visits * 350.0 + 
                        prescriptions_count * 150.0)
    
    # Random variation
    noise = np.random.normal(0, 1500, size=num_rows)
    
    annual_medical_cost = base_cost + age_cost + bmi_cost + smoker_cost + smoker_bmi_interaction + chronic_cost + utilization_cost + noise
    annual_medical_cost = np.clip(annual_medical_cost, 1200.0, 180000.0) # clip negative costs and cap extremely high costs
    
    # Risk score for Classification Label (will drop the risk_score itself)
    risk_score = (
        (age * 0.04) + 
        ((bmi - 25) * 0.08) + 
        (np.where(smoker == "Yes", 4.0, 0.0)) + 
        (diabetes * 2.5) + 
        (hypertension * 2.0) + 
        (heart_disease * 4.5) + 
        (kidney_disease * 3.5) + 
        (copd * 2.5) + 
        (hospital_visits * 0.8) + 
        (emergency_visits * 1.2) +
        (stress_level * 0.1) -
        (active_hours * 0.3)
    )
    
    # Determine risk threshold to get ~22% high risk patients
    threshold = np.percentile(risk_score, 78)
    is_high_risk = (risk_score > threshold).astype(int)
    
    # Put it all together
    df = pd.DataFrame({
        "Age": age,
        "Gender": gender,
        "Income": np.round(income, 2),
        "Children": children,
        "Region": region,
        "Employment_Status": employment_status,
        "Marital_Status": marital_status,
        "BMI": np.round(bmi, 2),
        "Smoker": smoker,
        "Alcohol": alcohol,
        "Exercise_Frequency": exercise_frequency,
        "Diet_Quality": diet_quality,
        "Sleep_Hours": np.round(sleep_hours, 1),
        "Stress_Level": stress_level,
        "Water_Intake": np.round(water_intake, 1),
        "Active_Hours": np.round(active_hours, 1),
        "Smokeless_Tobacco": smokeless_tobacco,
        "Diabetes": diabetes,
        "Hypertension": hypertension,
        "Heart_Disease": heart_disease,
        "Asthma": asthma,
        "Kidney_Disease": kidney_disease,
        "COPD": copd,
        "Joint_Pain": joint_pain,
        "Allergies": allergies,
        "Anxiety_Depression": anxiety_depression,
        "Thyroid_Issue": thyroid_issue,
        "SystolicBP": systolic_bp,
        "DiastolicBP": diastolic_bp,
        "Cholesterol": cholesterol,
        "BloodSugar": blood_sugar,
        "HbA1c": np.round(hba1c, 2),
        "Family_History_Diabetes": family_history_diabetes,
        "Family_History_Hypertension": family_history_hypertension,
        "Family_History_HeartDisease": family_history_heart_disease,
        "Hospital_Visits": hospital_visits,
        "Emergency_Visits": emergency_visits,
        "Specialist_Visits": specialist_visits,
        "Telehealth_Visits": telehealth_visits,
        "Preventative_Visits": preventative_visits,
        "Prescriptions_Count": prescriptions_count,
        "Prior_Authorizations": prior_authorizations,
        "Denials_Count": denials_count,
        "Adherence_Score": np.round(adherence_score, 3),
        "Satisfaction_Score": satisfaction_score,
        "Vaccination_Status": vaccination_status,
        "Insurance_Plan": insurance_plan,
        "Out_of_Pocket_Max": np.round(out_of_pocket_max, 2),
        "Deductible": np.round(deductible, 2),
        "Copay": np.round(copay, 2),
        "Urban_Rural": urban_rural,
        "Work_Life_Balance": work_life_balance,
        "Annual_Medical_Cost": np.round(annual_medical_cost, 2),
        "is_high_risk": is_high_risk
    })
    
    # Save the file
    os.makedirs("/Users/hemachandra/Documents/ML_capstone/data", exist_ok=True)
    filepath = "/Users/hemachandra/Documents/ML_capstone/data/insurance_extended.csv"
    df.to_csv(filepath, index=False)
    print(f"Dataset saved to {filepath}. Shape: {df.shape}")
    return df

if __name__ == "__main__":
    generate_dataset()
