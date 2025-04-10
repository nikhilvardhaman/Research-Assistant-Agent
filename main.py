import sys
print("sys",sys.executable)
import streamlit as st
import home
import ideation
import research_collection3


PAGES = {
    "Home": home,
    "Ideation": ideation,
    "Resource Collection": research_collection3
}  # Dictionary to map page names to their corresponding modules

def main():
    st.sidebar.title('Navigation')  # Setting the title of the sidebar for navigation
    selection = st.sidebar.radio("Go to", list(PAGES.keys()))  # Creating a radio button selection in the sidebar to choose different pages

    page = PAGES[selection]  # Getting the selected page from the PAGES dictionary (by default it will be Home as it is the first in the list)
    page.app()  # Calling the app function of the selected page module

if __name__ == "__main__":
    main()
