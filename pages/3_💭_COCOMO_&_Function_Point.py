import streamlit as st

# Calculation of TDI
def calc_TDI():
    TDI=14
    influence=st.selectbox(label="Enter the degree of influence of the function:", options=['Incidental', 'Moderate', 'Average', 'Significant', 'Essential', 'n/a'])
    influence=influence.lower()
    degree_of_influence={"n/a":0, "Incidental":1, "Moderate":2, "Average":3, "Significant":4, "Essential":5}

    if influence in degree_of_influence:
        TDI = TDI * degree_of_influence[influence]

    return TDI

# Calculation of UFP
def calc_UFP():
    attribute= [[3, 4, 6], [4, 5, 7], [3, 4, 6], [7, 10, 15], [5, 7, 10]]
    complexity_score={"Simple":0, "Average": 1, "Complex": 2}
    f_type_score={"Input": 0,"Output":1, "Inquiry":2, "ILF": 3, "ELF": 4}
    data=[]
    UFP=0

    while True:
        file_type=st.selectbox(label="Enter file type:", options=['Input', 'Output', 'ILF', 'ELF', 'Inquiry'])
        n_files=st.number_input("Enter number of files:")
        complexity=st.selectbox(label="Enter its complexity:", options=['Simple', 'Average', 'Complex'])

        data.append({
            "file_type": file_type,
            "complexity": complexity,
            "n_file": n_files
        })

        choice=st.selectbox(label="Choose file type:", options=['Yes','No'])
        if choice=="No":
            break

    for file in data:
        st.write(f"{file['n_file']} {file['file_type']} File that are {file['complexity']}")

    #print("\n")
    length=len(data)
    for i in range (length):
        file_type = data[i]['file_type']
        complexity_type=data[i]['complexity']

        # print(f"file_type: {file_type}")
        # print(f"complexity_type: {complexity_type}")

        if file_type in f_type_score:
            f_type=f_type_score[file_type]
            # print(f"f_type: {f_type}")

        if complexity_type in complexity_score:
            c_score=complexity_score[complexity_type]
            # print(f"c_score: {c_score}")

        UFP = UFP + (attribute[f_type][c_score] * data[i]['n_file'])
        # print(UFP)

    # print(f"Total UFP: {UFP}")
    return UFP

# Calculate Function Point
def FP():
    TDI=calc_TDI()
    CAF=0.65+0.01*TDI
    UFP=calc_UFP()
    FP=UFP*CAF
    st.write(f"The function point is: {FP}")

# COCOMO  Model Triggering Function
def contains_cocomo_keywords(text):
    keywords = ["cocomo", "COCOMO", "semi-detached", "organic", "embedded", "KLOC"]
    for keyword in keywords:
        if keyword.lower() in text.lower():
            return True
    return False

# Function Point Trigger
def check_function_point_analysis(text):
    text=text.lower()
    keywords=["function point"]
    for keyword in keywords:
        if keyword in text:
            return True
    return False

# Calculation of COCOMO NUmerical
def cocomo():
    kloc = st.number_input("Enter the lines of code in thousands: ")
    var = st.selectbox(label = "Which COCOMO Varient do you want to run calculations on:-", options=["Organic", "Semi Detached", "Embedded"])
    #valid = ["Organic", "Semi Detached", "Embedded"]
    #if var in valid:
    if var == "Organic":
        effort = 2.4*(kloc ** 1.05)
        devtime = 2.5 * (effort ** 0.38)
    elif var == "Semi Detached":
        effort = 3*(kloc ** 1.12)
        devtime = 2.5 * (effort ** 0.35)
    else:
        effort = 3.6*(kloc ** 1.20)
        devtime = 2.5 * (effort ** 0.32)
    try:
        wfr = effort/devtime
        prod = kloc/effort

        st.write(f"Effort: {effort}")
        st.write(f"Dev. time: {devtime}")
        st.write(f"Work force requirements (WFR): {wfr}")
        st.write(f"Productivity: {prod}")
    except Exception as e:
        st.write("KLOC is Zero!")
    
    #else:
        #print(var, "is not a valid option.")


if __name__ == '__main__':
    st.set_page_config(
        page_title="Document Chat App",
        page_icon="💬",
    )
    st.header("Ask A Question❔")
    prompt = st.selectbox("Enter your Question Type", options=['COCOMO','Function Point']) # this input should come from the user.
    # checking type of question and calling
    if contains_cocomo_keywords(prompt): # check the keywords as well, and you can change it at your discretion
        cocomo()
    elif check_function_point_analysis(prompt):
        try:
            FP()
        except Exception as e:
            st.write("No Input Given!")
    # else:
        # will go to the base model 