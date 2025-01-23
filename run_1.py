import os
from openai import OpenAI
from dotenv import load_dotenv

# 加载.env文件中的环境变量
load_dotenv()

# 初始化OpenAI客户端
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def evaluate_medical_text():
    # 在这里定义你的prompt模板
    prompt = """
    Task: Determine whether the following statement is logically entailed by the clinical trial report (CTR) section.
Type: Single,
Section_id: Eligibility,
Statement: 'Patients with Platelet count over 100,000/mm\u00ac\u00a8\u201a\u00e2\u2022, ANC <  1,700/mm\u00ac\u00a8\u201a\u00e2\u2022 and Hemoglobin between 4 to 5 grams per deciliter are eligible for the primary trial.'

Clinical Trial Report (CTR): 
{
    "Clinical Trial ID": "NCT00662129",
    "Intervention": [
        "INTERVENTION 1: ",
        "  Paclitaxel + Gemcitabine + Bevacizumab",
        "  Patients receive 125 mg/m^2 paclitaxel albumin-stabilized nanoparticle formulation IV over 30 minutes and 1000 mg/m^2 gemcitabine hydrochloride IV over 30 minutes on days 1 and 8, and 15 mg/kg bevacizumab IV over 30-90 minutes on day 1. Courses repeat every 21 days in the absence of disease progression or unacceptable toxicity."
    ],
    "Eligibility": [
        "DISEASE CHARACTERISTICS:",
        "  Histologically or cytologically confirmed infiltrating breast cancer",
        "  Clinical evidence of metastatic disease",
        "  Measurable disease, defined as at least one measurable lesion per RECIST criteria",
        "  No non-measurable disease only, defined as all other lesions, including small lesions (longest diameter < 2 cm) and truly non-measurable lesions, including any of the following:",
        "  Bone lesions",
        "  Leptomeningeal disease",
        "  Ascites",
        "  Pleural/pericardial effusion",
        "  Inflammatory breast disease",
        "  Lymphangitis cutis/pulmonis",
        "  Abdominal masses that are not confirmed and followed by imaging techniques",
        "  Cystic lesions",
        "  Patients with HER-2/neu positive tumors, must have received prior treatment with trastuzumab (Herceptin\u00ae) or have a contraindication for trastuzumab",
        "  No evidence of active brain metastasis, including leptomeningeal involvement, on MRI or CT scan",
        "  CNS metastasis controlled by prior surgery and/or radiotherapy allowed",
        "  Must be asymptomatic for  2 months with no evidence of progression prior to study entry",
        "  Hormone receptor status not specified",
        "  PATIENT CHARACTERISTICS:",
        "  Menopausal status not specified",
        "  Life expectancy  12 weeks",
        "  ECOG performance status 0-1",
        "  ANC  1,500/mm\u00b3",
        "  Platelet count  100,000/mm\u00b3",
        "  Hemoglobin  9.0 g/dL",
        "  AST and ALT  2.5 times upper limit of normal (ULN)",
        "  Alkaline phosphatase  2.5 times ULN",
        "  Total bilirubin  1.5 times ULN",
        "  Creatinine  1.5 mg/dL",
        "  Urine protein:creatinine ratio < 1 or urinalysis < 1+ protein",
        "  Patients discovered to have  1+ proteinuria at baseline must demonstrate 24-hour urine protein < 1 g",
        "  Not pregnant or nursing",
        "  Negative pregnancy test",
        "  Fertile patients must use effective contraception during and for 30 days after completion of study therapy",
        "  Able to complete questionnaires alone or with assistance",
        "  No peripheral neuropathy > grade 1",
        "  No history of allergy or hypersensitivity to albumin-bound paclitaxel, paclitaxel, gemcitabine hydrochloride, bevacizumab, albumin, drug product excipients, or chemically similar agents",
        "  No stage III or IV invasive, non-breast malignancy within the past 5 years",
        "  No other active malignancy, except nonmelanoma skin cancer or carcinoma in situ of the cervix",
        "  Patient must not be receiving other specific treatment for a prior malignancy",
        "  No uncontrolled hypertension (i.e., blood pressure [BP] > 160/90 mm Hg on  2 occasions at least 5 minutes apart)",
        "  Patients who have recently started or adjusted antihypertensive medications are eligible providing that BP is < 140/90 mm Hg on any new regimen for  3 different observations in  14 days",
        "  No bleeding diathesis or uncontrolled coagulopathy",
        "  No hemoptysis within the past 6 months",
        "  No prior arterial or venous thrombosis within the past 12 months",
        "  No history of cerebrovascular accident",
        "  No history of hypertensive crisis or hypertensive encephalopathy",
        "  No abdominal fistula or gastrointestinal perforation within the past 6 months",
        "  No serious non-healing wound, ulcer, or fracture",
        "  No clinically significant cardiac disease, defined as any of the following:",
        "  Congestive heart failure",
        "  Symptomatic coronary artery disease",
        "  Unstable angina",
        "  Cardiac arrhythmias not well controlled with medication",
        "  Myocardial infarction within the past 12 months",
        "  No comorbid systemic illnesses or other severe concurrent disease which, in the judgment of the investigator, would make the patient inappropriate for study entry or interfere significantly with the proper assessment of safety and toxicity of the prescribed regimens",
        "  PRIOR CONCURRENT THERAPY:",
        "  See Disease Characteristics",
        "  No prior chemotherapy for metastatic disease",
        "  May have received one prior adjuvant chemotherapy regimen",
        "  Prior neoadjuvant chemotherapy allowed",
        "  More than 6 months since prior adjuvant or neoadjuvant taxane (i.e., docetaxel or paclitaxel) therapy",
        "  Prior hormonal therapy in either adjuvant or metastatic setting allowed",
        "  More than 4 weeks since prior radiotherapy (except if to a non-target lesion only, or single dose radiation for palliation)",
        "  Prior radiotherapy to a target lesion is allowed provided there has been clear progression of the lesion since radiotherapy was completed",
        "  More than 4 weeks since prior cytotoxic chemotherapeutic agent or investigational drug",
        "  More than 2 weeks since prior and no concurrent acetylsalicylic acid, anticoagulants, or thrombolytic agents (except for once-daily 81 mg acetylsalicylic acid)",
        "  More than 6 weeks since prior major surgery, chemotherapy, or immunologic therapy",
        "  More than 1 week since prior minor surgery (e.g., core biopsy)",
        "  Placement of a vascular access device within 7 days is allowed",
        "  More than 3 months since prior neurosurgery",
        "  No concurrent treatment in a different clinical study in which investigational procedures are performed or investigational therapies are administered",
        "  Trials related to symptom management (Cancer Control) which do not employ hormonal treatments or treatments that may block the path of the targeted agents used in this study may be allowed"
    ],
    "Results": [
        "Outcome Measurement: ",
        "  6-month Progression-free Survival (PFS) Rate",
        "  The primary endpoint of this trial is the 6-month progression-free survival rate. A patient is considered to be a 6-month progression-free survivor if the patient is 6 months from registration without a documentation of disease progression (note, the patient need not be on study treatment at 6 months to be considered a success). The proportion of successes will be estimated by the number of successes divided by the total number of evaluable patients. Confidence intervals for the true success proportion will be calculated using the properties of the binomial distribution. Progression is defined using the RECIST Criteria, as at least a 20% increase in the sum of longest diameter (LD) of target lesions taking as reference the smallest sum LD recorded since the treatment started or the appearance of one or more new lesions, appearance of one or more new lesions, or unequivocal progression of existing non-target lesions.",
        "  Time frame: at 6 months",
        "Results 1: ",
        "  Arm/Group Title: Paclitaxel + Gemcitabine + Bevacizumab",
        "  Arm/Group Description: Patients receive 125 mg/m^2 paclitaxel albumin-stabilized nanoparticle formulation IV over 30 minutes and 1000 mg/m^2 gemcitabine hydrochloride IV over 30 minutes on days 1 and 8, and 15 mg/kg bevacizumab IV over 30-90 minutes on day 1. Courses repeat every 21 days in the absence of disease progression or unacceptable toxicity.",
        "  Overall Number of Participants Analyzed: 48",
        "  Measure Type: Number",
        "  Unit of Measure: proportion of patients progression-free  0.792        (0.647 to 0.882)"
    ],
    "Adverse Events": [
        "Adverse Events 1:",
        "  Total: 20/49 (40.82%)",
        "  Febrile neutropenia 1/49 (2.04%)",
        "  Hemoglobin decreased 3/49 (6.12%)",
        "  Constipation 1/49 (2.04%)",
        "  Diarrhea 3/49 (6.12%)",
        "  Mucositis oral 1/49 (2.04%)",
        "  Nausea 3/49 (6.12%)",
        "  Oral cavity fistula 1/49 (2.04%)",
        "  Vomiting 2/49 (4.08%)",
        "  Fatigue 3/49 (6.12%)",
        "  Fever 2/49 (4.08%)",
        "  Catheter related infection 1/49 (2.04%)",
        "  Infection 1/49 (2.04%)"
    ]
}

If the statement is true based on the section, return 'Entailment'. Do not include any other textual explanations
If the statement is false, return 'Contradiction'. Do not include any other textual explanations

What it the answer? Entailment or Contradiction？
    """
    
    try:
        # 调用API
        response = client.chat.completions.create(
            model="qwen-turbo",  
            messages=[
                {"role": "system", "content": "You are a medical expert specialized in analyzing medical texts."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=10
        )
        
        # 获取并返回结果
        result = response.choices[0].message.content.strip()
        return result
    
    except Exception as e:
        return f"发生错误: {str(e)}"

def main():
    # 在这里输入你要测试的医学文本
    
    
    # 获取分析结果
    result = evaluate_medical_text()
    
    # 打印结果
    print("\n=== 分析结果 ===")
    print(result)

if __name__ == "__main__":
    main()
