import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

model = pickle.load(open("model.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

teams = {
    "Mumbai Indians": {"color": "#004BA0", "logo": "https://upload.wikimedia.org/wikipedia/en/2/25/Mumbai_Indians_Logo.svg"},
    "Chennai Super Kings": {"color": "#FDB913", "logo": "https://upload.wikimedia.org/wikipedia/en/2/2e/Chennai_Super_Kings_Logo.svg"},
    "Royal Challengers Bangalore": {"color": "#D71920", "logo": "https://upload.wikimedia.org/wikipedia/en/4/4a/Royal_Challengers_Bangalore_Logo.svg"},
    "Kolkata Knight Riders": {"color": "#3A225D", "logo": "https://upload.wikimedia.org/wikipedia/en/4/4b/Kolkata_Knight_Riders_Logo.svg"},
    "Delhi Capitals": {"color": "#00008B", "logo": "https://upload.wikimedia.org/wikipedia/en/2/2f/Delhi_Capitals.svg"},
    "Kings XI Punjab": {"color": "#ED1B24", "logo": "https://upload.wikimedia.org/wikipedia/en/d/d4/Kings_XI_Punjab_logo.svg"},
    "Sunrisers Hyderabad": {"color": "#FF822A", "logo": "https://upload.wikimedia.org/wikipedia/en/8/81/Sunrisers_Hyderabad.svg"},
    "Rajasthan Royals": {"color": "#EA1A85", "logo": "https://upload.wikimedia.org/wikipedia/en/6/60/Rajasthan_Royals_Logo.svg"}
}

st.set_page_config(page_title="IPL Predictor", page_icon="🏏")

st.markdown("<h1 style='text-align:center;'>🏏 IPL Win Predictor</h1>", unsafe_allow_html=True)

# Team selection
col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox("Batting Team", list(teams.keys()))
    st.image(teams[batting_team]["logo"], width=120)

with col2:
    bowling_team = st.selectbox("Bowling Team", list(teams.keys()))
    st.image(teams[bowling_team]["logo"], width=120)

# Inputs
runs_left = st.number_input("Runs Left", min_value=0)
balls_left = st.number_input("Balls Left", min_value=0)
wickets_left = st.number_input("Wickets Left", min_value=0, max_value=10)
total_runs = st.number_input("Current Score", min_value=0)
overs = st.number_input("Overs Completed", min_value=0.0)

if st.button("Predict"):

    crr = total_runs / overs if overs > 0 else 0
    rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

    input_dict = {
        'runs_left': runs_left,
        'balls_left': balls_left,
        'wickets_left': wickets_left,
        'crr': crr,
        'rrr': rrr
    }

    for col in columns:
        if col.startswith("batting_team_"):
            input_dict[col] = 1 if col == f"batting_team_{batting_team}" else 0
        elif col.startswith("bowling_team_"):
            input_dict[col] = 1 if col == f"bowling_team_{bowling_team}" else 0

    input_df = pd.DataFrame([input_dict])
    input_df = input_df.reindex(columns=columns, fill_value=0)

    prob = model.predict_proba(input_df)[0]

    st.markdown("##  Prediction")

    col3, col4 = st.columns(2)

    with col3:
        st.markdown(f"""
        <div style="background-color:{teams[batting_team]['color']};
                    padding:20px;border-radius:15px;color:white;text-align:center;">
            <img src="{teams[batting_team]['logo']}" width="80"><br>
            <h3>{batting_team}</h3>
            <h2>{round(prob[1]*100,2)}%</h2>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div style="background-color:{teams[bowling_team]['color']};
                    padding:20px;border-radius:15px;color:white;text-align:center;">
            <img src="{teams[bowling_team]['logo']}" width="80"><br>
            <h3>{bowling_team}</h3>
            <h2>{round(prob[0]*100,2)}%</h2>
        </div>
        """, unsafe_allow_html=True)

    st.progress(int(prob[1]*100))

    # Graph
    overs_list = list(range(1, 21))
    prob_list = []

    for o in overs_list:
        temp_balls_left = 120 - o*6
        if temp_balls_left <= 0:
            continue

        temp_rrr = (runs_left * 6) / temp_balls_left if temp_balls_left > 0 else 0

        temp_input = {
            'runs_left': runs_left,
            'balls_left': temp_balls_left,
            'wickets_left': wickets_left,
            'crr': crr,
            'rrr': temp_rrr
        }

        for col in columns:
            if col.startswith("batting_team_"):
                temp_input[col] = 1 if col == f"batting_team_{batting_team}" else 0
            elif col.startswith("bowling_team_"):
                temp_input[col] = 1 if col == f"bowling_team_{bowling_team}" else 0

        temp_df = pd.DataFrame([temp_input])
        temp_df = temp_df.reindex(columns=columns, fill_value=0)

        p = model.predict_proba(temp_df)[0][1]
        prob_list.append(p)

    plt.figure()
    plt.plot(overs_list[:len(prob_list)], prob_list)
    plt.title("Win Probability Trend")
    plt.xlabel("Overs")
    plt.ylabel("Probability")
    plt.grid()

    st.pyplot(plt)