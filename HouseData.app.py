import streamlit as st
import pickle
import numpy as np

# Load the saved model
model = pickle.load(open(r'/Users/aviswe/VS-code/Simple_Linear_Regressions/linear_regression_model.pkl', 'rb'))

# Set the title of the Streamlit app
st.title("Housing Sales Prediction App")

# Add a brief description
st.write("This app predicts the housing sale based on space using a simple linear regression model.")

# Add input widget for user to enter years of experience
#years_experience = st.number_input("space:", min_value=290.0, max_value=13540.0, value=1.0, step=100.0)
space = st.number_input("space:", min_value=290.0, max_value=13540.0, value=290.0, step=100.0)

# When the button is clicked, make predictions
if st.button("Predict Price"):
    # Make a prediction using the trained model
    #space_input = np.array([[space]])  # Convert the input to a 2D array for prediction
    space_input = np.array([[space]])

    prediction = model.predict(space_input)
   
    # Display the result
    st.success(f"The predicted salary for {space} space is: ${float(prediction[0]):,.2f}")
    #st.success(f"The predicted salary for {space} space is: ${float(prediction[0]):,.2f}")

   
# Display information about the model
st.write("The model was trained using a dataset of salaries and space.built model by Venkata Avija")