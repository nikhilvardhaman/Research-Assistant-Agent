import streamlit as st

def app():
    st.title("Welcome to Your AI Research Assistant")
    st.subheader("Your AI-powered companion for accelerating and simplifying your research journey.")

    st.write("""
    This AI Research Assistant is designed to assist researchers, students, and professionals in navigating 
    the complexities of research and idea generation. With core functionalities spread across different pages, 
    it is your ultimate tool for exploring new ideas and gathering relevant information swiftly and effectively.
    """)

    st.header("Capabilities of the AI Research Assistant")
    st.write("""
    **1. Speed and Efficiency:**  
    The AI Research Assistant can rapidly generate ideas, explore new concepts, and collect research materials, 
    saving you countless hours of manual work. It streamlines the research process, allowing you to focus on 
    deeper analysis and innovation.

    **2. Simplification of Complex Processes:**  
    Whether you're ideating new research topics or compiling existing literature, this assistant breaks down 
    complex tasks into manageable steps. It provides a clear path from conceptualization to research execution, 
    making the learning curve less steep.

    **3. AI-Powered Insights:**  
    Leverage the power of AI to gain insights that might not be immediately apparent. The assistant helps in 
    connecting dots, identifying trends, and suggesting new avenues of exploration based on your inputs.
    """)

    st.header("How It Works")
    st.write("""
    **Ideating:**  
    Use the Ideating page to brainstorm and generate new research ideas. The AI will assist in expanding upon 
    your initial thoughts, offering related concepts, potential research questions, and identifying gaps in the 
    current literature.

    **Research Collection:**  
    On the Research Collection page, you can input specific topics or questions. The AI will help you gather 
    relevant academic papers, articles, and data sources, organizing them in a coherent way for your review and 
    further study.
    """)
