import streamlit as st
from langchain import LLMChain, PromptTemplate
from langchain.chains import SimpleSequentialChain
from langchain_together import ChatTogether

# Initialize the LLM with the provided API key and model
llm = ChatTogether(api_key=st.secrets['togetherai_apikey'], temperature=0.0, model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo")

# Define a prompt template for analyzing and expanding on the topic
research_prompt = PromptTemplate(
    input_variables=["topic"],
    template=(
        "You are a research assistant. The user has asked the following research question or provided this topic: '{topic}'. "
        "Please analyze the topic, explain it back to the user, and suggest 4 additional related questions and ideas that "
        "could help expand the research. Generate a list of search queries (including the original one) or topics that could be helpful in researching about the topic."
    )
)

# Create an LLMChain using the defined prompt template
research_chain = LLMChain(
    llm=llm,
    prompt=research_prompt
)

# Simple Sequential Chain to handle the processing
sequential_chain = SimpleSequentialChain(
    chains=[research_chain],
    verbose=True
)

def generate_research_insights(topic):
    # Run the sequential chain with the provided topic
    response = sequential_chain.run(topic)
    return response

def app():
    st.title("Ideating: Generate and Expand Research Ideas")
    st.subheader("Kickstart your research with AI-powered brainstorming.")

    st.write("""
    Input your initial ideas, keywords, or topics below, and the AI will help you expand on them by generating 
    related concepts, questions, and potential research directions.
    """)

    # Input area for user's ideas
    topic = st.text_area("Enter your research idea or topic:")

    if st.button("Generate Ideas and Insights"):
        if topic:
            st.write(f"**Your Idea:** {topic}")
            with st.spinner('Generating insights...'):  ## see now this is something new for you adding feedback onpage
                # Generate research insights using the LLM
                insights = generate_research_insights(topic)
            st.write("**AI-Generated Suggestions and Insights:**")
            st.write(insights)
        else:
            st.warning("Please enter a topic or idea to generate insights.")

if __name__ == "__main__":
    app()


### as an experiment you can make this an chatbot, with turn by turn coversation to discuss and improve the ideas 