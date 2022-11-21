from pydantic import BaseModel


class DefaultPrediction(BaseModel):
  name_contract_type: str
  code_gender: str
  flag_own_car: str
  flag_own_realty: str
  cnt_children: int
  amt_income_total: float
  amt_credit: float
  amt_annuity: float
  amt_goods_price: float
  name_type_suite: str
  name_income_type: str
  name_education_type: str
  name_family_status: str
  name_housing_type:str
  region_population_relative: float
  flag_mobil: int
  occupation_type: str
  cnt_fam_members: int
  region_rating_client_w_city: int
  weekday_appr_process_start: int
  hour_appr_process_start: int
  organization_type: str
  ext_source_1: float
  ext_source_2: float
  ext_source_3: float
  def_30_cnt_social_circle: int
  obs_60_cnt_social_circle: int
  amt_req_credit_bureau_year: int
  submitted_addit_docs: float
  prop_of_unmatched_contact_perm_work_addr: float
  prop_of_provided_living_place_info: float
  prop_of_provided_phone_email_info: float
  age_years: float
  last_reg_change_in_years: float
  last_phone_change_in_years: float
  employment_years: float
  id_published_years: float
  living_in_owning_realty: int
  prev_approved_hc: float
  prev_amt_credit_mean_hc: float
  prev_amt_annuity_mean_hc: float
  prev_min_loan_term_hc: float
  prev_max_loan_term_hc: float
  prev_days_from_last_approval_hc: float
  approved_hc_name_contract_type_Cash_loans_normalized: float
  approved_hc_name_contract_type_Consumer_loans_normalized: float
  approved_hc_name_contract_type_Revolving_loans_normalized: float
  approved_hc_name_client_type_New_normalized: float
  approved_hc_name_client_type_Refreshed_normalized: float
  approved_hc_name_client_type_Repeater_normalized: float
  approved_hc_name_client_type_XNA_normalized: float
  approved_hc_name_yield_group_XNA_normalized: float
  approved_hc_name_yield_group_high_normalized: float
  approved_hc_name_yield_group_low_action_normalized: float
  approved_hc_name_yield_group_low_normal_normalized: float
  approved_hc_name_yield_group_middle_normalized: float
  total_appl_hc: float
  total_rejected_appl_hc: float
  prev_reject_amt_appl_mean_hc: float
  mean_delayed_days_per_all_loans_hc: float
  total_delayed_days_per_all_loans_hc: float
  total_future_instalments_by_other_credts_hc: float
  total_active_credits_hc: float
  total_defaults_previous_credits_hc: float
  min_instal_amt_per_prev_credit_hc: float
  max_instal_amt_per_prev_credit_hc: float
  mean_instal_amt_per_prev_credit_hc: float
  total_instalments_amt_prev_credit_hc: float
  total_credit_cards_hc: float
  max_total_per_1cred_card_limit_hc: float
  mean_total_per_1cred_card_limit_hc: float
  mean_drawings_per_1cred_card_hc: float
  mean_credit_card_payment_hc: float
  total_default_days_per_all_cred_cards_hc: float
  total_defaults_credit_cards_hc: float
  total_credits_bureau: float
  total_credit_types_count_bureau: float
  mean_day_overdue_per_card_bureau: float
  mean_debt_per_card_bureau: float
  mean_credit_per_card_bureau: float
  avg_credits_prolonged_bureau: float
  bureau_credit_active_Active_normalized: float
  bureau_credit_active_Bad_debt_normalized: float
  bureau_credit_active_Closed_normalized: float
  bureau_credit_active_Sold_normalized: float


name_contract_type_dict = {
  "cash_loans": "cash_loans",
  "revolving_loans": "revolving_loans",
  }

code_gender_dict = {
  "f":"f",
  "m":"m",
  "xna":"xna"
}


flag_own_car_dict = {
  "n": "n", 
  "y": "y",
}


flag_own_realty_dict = {
  "n": "n", 
  "y": "y",
}

name_type_suite_dict = {
  "unaccompanied":"unaccompanied",
  "family":"family",
  "other":"other",
  "unknown":"unknown",
}

name_income_type_dict = {
  "working":"working",
  "commercial_associate": "commercial_associate",
  "pensioner": "pensioner",
  "state_servant":"state_servant",
  "unemployed":"unemployed",
  "student": "student",
  "businessman":"businessman",
  "maternity_leave":"maternity_leave",
}

name_education_type_dict = {
  "secondary_secondary_special":"secondary_secondary_special",
  "higher_education":"higher_education",
  "incomplete_higher":"incomplete_higher",
  "lower_secondary":"lower_secondary",
  "academic_degree":"academic_degree",
}

name_family_status_dict = {
"married":"married",
"single_not_married":"single_not_married",
"civil_marriage":"civil_marriage",
"separated":"separated",
"widow":"widow",
"unknown":"unknown",
}

name_housing_type_dict = {
  "house_apartment":"house_apartment",
  "with_parents":"with_parents",
  "municipal_apartment":"municipal_apartment",
  "rented_apartment":"rented_apartment",
  "office_apartment":"office_apartment",
  "co-op_apartment":"co-op_apartment",
}

occupation_type_dict = {
  "unknown":"unknown",
  "laborers":"laborers",
  "sales_staff":"sales_staff",
  "core_staff":"core_staff",
  "managers":"managers",
  "drivers":"drivers",
  "high_skill_tech_staff":"high_skill_tech_staff",
  "accountants":"accountants",
  "medicine_staff":"medicine_staff",
  "security_staff":"security_staff",
  "cooking_staff":"cooking_staff",
  "cleaning_staff":"cleaning_staff",
  "private_service_staff":"private_service_staff",
  "low-skill_laborers":"low-skill_laborers",
  "waiters/barmen_staff":"waiters/barmen_staff",
  "secretaries":"secretaries",
  "realty_agents":"realty_agents",
  "hr_staff":"hr_staff",
  "it_staff":"it_staff",
}

organization_type_dict = {
"business_entity_type_3":"business_entity_type_3",
"xna":"xna",
"self-employed":"self-employed",
"other":"other",
"medicine":"medicine",
"business_entity_type_2":"business_entity_type_2",
"government":"government",
"school":"school",
"trade:_type_7":"trade:_type_7",
"kindergarten":"kindergarten",
"construction":"construction",
"business_entity_type_1":"business_entity_type_1",
"transport:_type_4":"transport:_type_4",
"trade:_type_3":"trade:_type_3",
"industry:_type_9":"industry:_type_9",
"industry:_type_3":"industry:_type_3",
"security":"security",
"housing":"housing",
"industry:_type_11":"industry:_type_11",
"military":"military",
"agriculture":"agriculture",
"bank":"bank",
"police":"police",
"transport:_type_2":"transport:_type_2",
"postal":"postal",
"security_ministries":"security_ministries",
"trade:_type_2":"trade:_type_2",
"restaurant":"restaurant",
"services":"services",
"university":"university",
"industry:_type_7":"industry:_type_7",
"transport:_type_3":"transport:_type_3",
"industry:_type_1":"industry:_type_1",
"hotel":"hotel",
"electricity":"electricity",
"industry:_type_4":"industry:_type_4",
"industry:_type_5":"industry:_type_5",
"trade:_type_6":"trade:_type_6",
"insurance":"insurance",
"telecom":"telecom",
"emergency":"emergency",
"industry:_type_2":"industry:_type_2",
"advertising":"advertising",
"realtor":"realtor",
"culture":"culture",
"industry:_type_12":"industry:_type_12",
"trade:_type_1":"trade:_type_1",
"legal_services":"legal_services",
"mobile":"mobile",
"cleaning":"cleaning",
"transport:_type_1":"transport:_type_1",
"industry:_type_6":"industry:_type_6",
"industry:_type_10":"industry:_type_10",
"religion":"religion",
"trade:_type_4":"trade:_type_4",
"industry:_type_13":"industry:_type_13",
"trade:_type_5":"trade:_type_5",
"industry:_type_8":"industry:_type_8",
}
