import joblib
import uvicorn, os
import pandas as pd
from lightgbm import LGBMClassifier
from fastapi import FastAPI, HTTPException
from classes import *

app = FastAPI()

default_predictor = joblib.load("model_lgbm.joblib")


@app.get("/")
def home():
    return {"text": "Default in loan predictor"}

@app.post("/default_prediction")
async def create_application(default_pred: DefaultPrediction):

    default_df = pd.DataFrame()

    if default_pred.name_contract_type not in name_contract_type_dict:
        raise HTTPException(status_code=404, detail="Contract type not found")

    if default_pred.code_gender not in code_gender_dict:
        raise HTTPException(status_code=404, detail="Please insert normal type of gender")

    if default_pred.flag_own_car not in flag_own_car_dict:
            raise HTTPException(status_code=404, detail="So, do you have a car?")

    if default_pred.flag_own_realty not in flag_own_realty_dict:
            raise HTTPException(status_code=404, detail="So, do you have any real estate?")

    if default_pred.name_type_suite not in name_type_suite_dict:
            raise HTTPException(status_code=404, detail="Type of suit not found.")

    if default_pred.name_income_type not in name_income_type_dict:
            raise HTTPException(status_code=404, detail="Type of income not found.")

    if default_pred.name_education_type not in name_education_type_dict:
            raise HTTPException(status_code=404, detail="Type of education not found.")

    if default_pred.name_family_status not in name_family_status_dict:
            raise HTTPException(status_code=404, detail="So, what is your family status?")

    if default_pred.name_housing_type not in name_housing_type_dict:
            raise HTTPException(status_code=404, detail="Where do you live in?")

    if default_pred.occupation_type not in occupation_type_dict:
            raise HTTPException(status_code=404, detail="Occupation type not found.")

    if default_pred.organization_type not in organization_type_dict:
            raise HTTPException(status_code=404, detail="Type of organization not found.")

    default_df["name_contract_type"] = [default_pred.name_contract_type]
    default_df["code_gender"] = [default_pred.code_gender]
    default_df["flag_own_car"] = [default_pred.flag_own_car]
    default_df["flag_own_realty"] = [default_pred.flag_own_realty]
    default_df["cnt_children"] = [default_pred.cnt_children]
    default_df["amt_income_total"] = [default_pred.amt_income_total]
    default_df["amt_credit"] = [default_pred.amt_credit]
    default_df["amt_annuity"] = [default_pred.amt_annuity]
    default_df["amt_goods_price"] = [default_pred.amt_goods_price]
    default_df["name_type_suite"]= [default_pred.name_type_suite]
    default_df["name_income_type"]= [default_pred.name_income_type]
    default_df["name_education_type"] = [default_pred.name_education_type]
    default_df["name_family_status"] = [default_pred.name_family_status]
    default_df["name_housing_type"] = [default_pred.name_housing_type]
    default_df["region_population_relative"] = [default_pred.region_population_relative]
    default_df["flag_mobil"] = [default_pred.flag_mobil]
    default_df["occupation_type"] = [default_pred.occupation_type]
    default_df["cnt_fam_members"] = [default_pred.cnt_fam_members]
    default_df["region_rating_client_w_city"] = [default_pred.region_rating_client_w_city]
    default_df["weekday_appr_process_start"] = [default_pred.weekday_appr_process_start]
    default_df["hour_appr_process_start"] = [default_pred.hour_appr_process_start]
    default_df["organization_type"] = [default_pred.organization_type]
    default_df["ext_source_1"]=[default_pred.ext_source_1]
    default_df["ext_source_2"]=[default_pred.ext_source_2]
    default_df["ext_source_3"]=[default_pred.ext_source_3]
    default_df["def_30_cnt_social_circle"]=[default_pred.def_30_cnt_social_circle]
    default_df["obs_60_cnt_social_circle"]=[default_pred.obs_60_cnt_social_circle]
    default_df["amt_req_credit_bureau_year"]=[default_pred.amt_req_credit_bureau_year]
    default_df["submitted_addit_docs"]=[default_pred.submitted_addit_docs]
    default_df["prop_of_unmatched_contact_perm_work_addr"]=[default_pred.prop_of_unmatched_contact_perm_work_addr]
    default_df["prop_of_provided_living_place_info"]=[default_pred.prop_of_provided_living_place_info]
    default_df["prop_of_provided_phone_email_info"]=[default_pred.prop_of_provided_phone_email_info]
    default_df["age_years"]=[default_pred.age_years]
    default_df["last_reg_change_in_years"]=[default_pred.last_reg_change_in_years]
    default_df["last_phone_change_in_years"]=[default_pred.last_phone_change_in_years]
    default_df["employment_years"]=[default_pred.employment_years]
    default_df["id_published_years"]=[default_pred.id_published_years]
    default_df["living_in_owning_realty"]=[default_pred.living_in_owning_realty]
    default_df["prev_approved_hc"]=[default_pred.prev_approved_hc]
    default_df["prev_amt_credit_mean_hc"]=[default_pred.prev_amt_credit_mean_hc]
    default_df["prev_amt_annuity_mean_hc"]=[default_pred.prev_amt_annuity_mean_hc]
    default_df["prev_min_loan_term_hc"]=[default_pred.prev_min_loan_term_hc]
    default_df["prev_max_loan_term_hc"]=[default_pred.prev_max_loan_term_hc]
    default_df["prev_days_from_last_approval_hc"]=[default_pred.prev_days_from_last_approval_hc]
    default_df["approved_hc_name_contract_type_Cash_loans_normalized"]=[default_pred.approved_hc_name_contract_type_Cash_loans_normalized]
    default_df["approved_hc_name_contract_type_Consumer_loans_normalized"]=[default_pred.approved_hc_name_contract_type_Consumer_loans_normalized]
    default_df["approved_hc_name_contract_type_Revolving_loans_normalized"]=[default_pred.approved_hc_name_contract_type_Revolving_loans_normalized]
    default_df["approved_hc_name_client_type_New_normalized"]=[default_pred.approved_hc_name_client_type_New_normalized]
    default_df["approved_hc_name_client_type_Refreshed_normalized"]=[default_pred.approved_hc_name_client_type_Refreshed_normalized]
    default_df["approved_hc_name_client_type_Repeater_normalized"]=[default_pred.approved_hc_name_client_type_Repeater_normalized]
    default_df["approved_hc_name_client_type_XNA_normalized"]=[default_pred.approved_hc_name_client_type_XNA_normalized]
    default_df["approved_hc_name_yield_group_XNA_normalized"]=[default_pred.approved_hc_name_yield_group_XNA_normalized]
    default_df["approved_hc_name_yield_group_high_normalized"]=[default_pred.approved_hc_name_yield_group_high_normalized]
    default_df["approved_hc_name_yield_group_low_action_normalized"]=[default_pred.approved_hc_name_yield_group_low_action_normalized]
    default_df["approved_hc_name_yield_group_low_normal_normalized"]=[default_pred.approved_hc_name_yield_group_low_normal_normalized]
    default_df["approved_hc_name_yield_group_middle_normalized"]=[default_pred.approved_hc_name_yield_group_middle_normalized]
    default_df["total_appl_hc"]=[default_pred.total_appl_hc]
    default_df["total_rejected_appl_hc"]=[default_pred.total_rejected_appl_hc]
    default_df["prev_reject_amt_appl_mean_hc"]=[default_pred.prev_reject_amt_appl_mean_hc]
    default_df["mean_delayed_days_per_all_loans_hc"]=[default_pred.mean_delayed_days_per_all_loans_hc]
    default_df["total_delayed_days_per_all_loans_hc"]=[default_pred.total_delayed_days_per_all_loans_hc]
    default_df["total_future_instalments_by_other_credts_hc"]=[default_pred.total_future_instalments_by_other_credts_hc]
    default_df["total_active_credits_hc"]=[default_pred.total_active_credits_hc]
    default_df["total_defaults_previous_credits_hc"]=[default_pred.total_defaults_previous_credits_hc]
    default_df["min_instal_amt_per_prev_credit_hc"]=[default_pred.min_instal_amt_per_prev_credit_hc]
    default_df["max_instal_amt_per_prev_credit_hc"]=[default_pred.max_instal_amt_per_prev_credit_hc]
    default_df["mean_instal_amt_per_prev_credit_hc"]=[default_pred.mean_instal_amt_per_prev_credit_hc]
    default_df["total_instalments_amt_prev_credit_hc"]=[default_pred.total_instalments_amt_prev_credit_hc]
    default_df["total_credit_cards_hc"]=[default_pred.total_credit_cards_hc]
    default_df["max_total_per_1cred_card_limit_hc"]=[default_pred.max_total_per_1cred_card_limit_hc]
    default_df["mean_total_per_1cred_card_limit_hc"]=[default_pred.mean_total_per_1cred_card_limit_hc]
    default_df["mean_drawings_per_1cred_card_hc"]=[default_pred.mean_drawings_per_1cred_card_hc]
    default_df["mean_credit_card_payment_hc"]=[default_pred.mean_credit_card_payment_hc]
    default_df["total_default_days_per_all_cred_cards_hc"]=[default_pred.total_default_days_per_all_cred_cards_hc]
    default_df["total_defaults_credit_cards_hc"]=[default_pred.total_defaults_credit_cards_hc]
    default_df["total_credits_bureau"]=[default_pred.total_credits_bureau]
    default_df["total_credit_types_count_bureau"]=[default_pred.total_credit_types_count_bureau]
    default_df["mean_day_overdue_per_card_bureau"]=[default_pred.mean_day_overdue_per_card_bureau]
    default_df["mean_debt_per_card_bureau"]=[default_pred.mean_debt_per_card_bureau]
    default_df["mean_credit_per_card_bureau"]=[default_pred.mean_credit_per_card_bureau]
    default_df["avg_credits_prolonged_bureau"]=[default_pred.avg_credits_prolonged_bureau]
    default_df["bureau_credit_active_Active_normalized"]=[default_pred.bureau_credit_active_Active_normalized]
    default_df["bureau_credit_active_Bad_debt_normalized"]=[default_pred.bureau_credit_active_Bad_debt_normalized]
    default_df["bureau_credit_active_Closed_normalized"]=[default_pred.bureau_credit_active_Closed_normalized]
    default_df["bureau_credit_active_Sold_normalized"]=[default_pred.bureau_credit_active_Sold_normalized]
    
    default_df["name_contract_type"] = default_df["name_contract_type"].astype("category")
    default_df["code_gender"] = default_df["code_gender"].astype("category")
    default_df["flag_own_car"] = default_df["flag_own_car"].astype("category")
    default_df["flag_own_realty"] = default_df["flag_own_realty"].astype("category")
    default_df["name_type_suite"] = default_df["name_type_suite"].astype("category")
    default_df["name_income_type"] = default_df["name_income_type"].astype("category")
    default_df["name_education_type"] = default_df["name_education_type"].astype("category")
    default_df["name_family_status"] = default_df["name_family_status"].astype("category")
    default_df["name_housing_type"] = default_df["name_housing_type"].astype("category")
    default_df["occupation_type"] = default_df["occupation_type"].astype("category")
    default_df["organization_type"] = default_df["organization_type"].astype("category")

    

    prediction = default_predictor.predict_proba(default_df).round(2)
    prediction = prediction[0][1]

    return {"probability of default": prediction}

# uvicorn main:app --reload
# uvicorn main:app --host 127.0.0.1 --port 8000
