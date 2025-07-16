## DynaCEL: Dynamic Cohort Ensemble Learning for Personalized and Real-Time Hemodynamic Management in Critical Care
Lingzhong Meng<sup>1</sup>, Jiangqiong Li<sup>2</sup>, Xiang Liu<sup>2</sup>, Yanhua Sun<sup>1,3</sup>, Zuotian Li<sup>2,4</sup>, Ameya D. Parab<sup>5</sup>, George Lu<sup>2</sup>, Aishwarya S. Budhkar<sup>5</sup>, Saravanan Kanakasabai<sup>6</sup>, David C. Adams<sup>1</sup>, Ziyue Liu<sup>2</sup>, Xuhong Zhang<sup>5</sup>, Jing Su<sup>2*</sup>

**Author affiliations:**

<sup>1</sup>Department of Anesthesia, Indiana University School of Medicine, Indianapolis, Indiana, USA 

<sup>2</sup>Department of Biostatistics and Health Data Science, Indiana University School of Medicine, Indianapolis, Indiana, USA

<sup>3</sup>Department of Anesthesiology, Nanjing Drum Tower Hospital, The Affiliated Hospital of Nanjing University Medical School, Nanjing, Jiangsu, China

<sup>4</sup>Department of Computer Graphic Technology, Polytechnic Institute, Purdue University, West Lafayette, Indiana, USA

<sup>5</sup>Department of Computer Science, Luddy School of Informatics, Computing, and Engineering, Indiana University Bloomington, Bloomington, Indiana, USA

<sup>6</sup>Clinical Research Systems, Enterprise Analytics, Indiana University Health, Indianapolis, Indiana, USA

<sup>*</sup>Corresponding Author: Jing Su, Department of Biostatistics and Health Data Science, Indiana University School of Medicine, Indianapolis, USA. Email: su1@iu.edu 

Effective hemodynamic management in the intensive care unit (ICU) requires individualized targets that adapt to rapidly changing clinical conditions. Current practice relies on fixed or empirical heart rate (HR) and systolic blood pressure (SBP) targets that often overlook patient variability. We present Dynamic Cohort Ensemble Learning (DynaCEL), a real-time modeling framework that recommends personalized HR and SBP targets by treating each moment after ICU admission as a distinct temporal cohort. Trained on the eICU database, DynaCEL was validated on the Indiana University Health (IUH) ICU and the MIMIC-IV cohorts, demonstrating robust performance across populations, time points, and model types. In the MIMIC-IV cohort, assuming target achievability, adherence to DynaCEL’s dynamic targets reduced 24-hour mortality from 0.85% to 0.04% compared with fixed targets (HR 80 bpm, SBP 120 mmHg), after adjustment for confounding via propensity score matching. DynaCEL also identified patient-specific safe zones and their evolution over time. Its predictive performance reached AUCs of 0.85–0.88 (IUH), 0.84–0.91 (eICU), and 0.83–0.89 (MIMIC-IV) using a 12-hour predictor window and 24-hour outcome window across multiple time points. Case studies further illustrated inter-individual and temporal variation in optimal targets. This novel big data-driven framework provides interpretable, dynamic, and personalized HR and BP targets, addressing a critical gap in ICU care. DynaCEL offers a scalable approach to precision hemodynamic management, with the potential to improve outcomes and reduce practice variability, although its clinical utility requires validation through prospective trials before real-world adoption.

<img width="358" height="562" alt="image" src="https://github.com/user-attachments/assets/0becf1b4-1e6e-4f5e-983a-82d4fb1b9523" />

**Figure: The DynaCEL framework: translating ICU big data into personalized, dynamic bedside hemodynamic management**

DynaCEL identifies real-time, individualized hemodynamic targets associated with the lowest acute mortality risk. It uses national ICU big data (e.g., eICU) to construct temporal cohorts representing patient populations at specific post-admission time points (e.g., T = 0, 18, 36 hours), each with distinct clinical characteristics and mortality profiles. For each cohort, a separate model is trained to learn HR and SBP targets linked to 24-hour mortality risk. These models form a temporal ensemble, enabling bedside deployment in external hospitals (e.g., IU Health, Beth Israel Deaconess Medical Center). For a new patient, DynaCEL recommends HR/SBP targets based on current clinical data, visualizes optimal targets on a mortality contour map, and displays the patient’s actual vs. target trajectory. It also provides alerts for deviations from recommended targets. DynaCEL is plug-and-play, supporting real-time, interpretable, and personalized hemodynamic management at the bedside.

DynaCEL, Dynamic Cohort Ensemble Learning; ICU, intensive care unit; HR, heart rate; SBP, systolic blood pressure; MOP, moment of prediction (time since ICU admission). 
