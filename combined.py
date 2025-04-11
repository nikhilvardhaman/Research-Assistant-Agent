import streamlit as st
import re
from langchain import LLMChain, PromptTemplate
from langchain.chains import SimpleSequentialChain
from langchain_together import ChatTogether
from langchain_community.tools import YouTubeSearchTool
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_community.tools.semanticscholar.tool import SemanticScholarQueryRun
from langchain_community.utilities.semanticscholar import SemanticScholarAPIWrapper
from langchain.agents import AgentExecutor, create_tool_calling_agent, initialize_agent, AgentType
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools.playwright.utils import create_async_playwright_browser
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain_community.tools.pubmed.tool import PubmedQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents import load_tools

# LLM
llm = ChatTogether(api_key=st.secrets['togetherai_apikey'], temperature=0.0, model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo")

# ------------------------- IDEATION -------------------------
research_prompt = PromptTemplate(
    input_variables=["topic"],
    template=(
        "You are a research assistant. The user has asked the following research question or provided this topic: '{topic}'. "
        "Please analyze the topic, explain it back to the user, and suggest 4 additional related questions and ideas that "
        "could help expand the research. Generate a list of search queries (including the original one) or topics that could be helpful in researching about the topic."
    )
)
research_chain = LLMChain(llm=llm, prompt=research_prompt)
sequential_chain = SimpleSequentialChain(chains=[research_chain], verbose=True)

def generate_research_insights(topic):
    return sequential_chain.run(topic)

def extract_topics_from_text(text):
    return re.findall(r"(?:\d\.|\-|\*)?\s*(?:\"|“)?(.+?)(?:\"|”)?(?:\n|$)", text)

# ------------------------- RESEARCH COLLECTION -------------------------

@tool
def youtube_transcript_tool(query: str) -> str:
    """Returns the transcript and YouTube URL of the top video for a given query."""
    tool = YouTubeSearchTool(language="en")
    yt_url = tool.run(f"{query},1")
    match = re.search(r"v=([^&\s]+)", yt_url)
    if not match:
        return "No video ID found in search result."
    video_id = match.group(1)
    video_link = f"https://www.youtube.com/watch?v={video_id}"
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = ' '.join([t['text'] for t in transcript])
        return f"Transcript from video: {video_link}\n\n{transcript_text}"
    except Exception as e:
        return f"Error fetching transcript from video: {video_link}\nReason: {str(e)}"

yt_tool = [youtube_transcript_tool]
youtube_agent_chain = initialize_agent(yt_tool, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, handle_parse_errors=True, max_iterations=3)

semantic_scholar_api = SemanticScholarAPIWrapper(doc_content_chars_max=1000, top_k_results=5)
semantic_scholar_tool = SemanticScholarQueryRun(api_wrapper=semantic_scholar_api)
semantic_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert researcher."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])
semantic_scholar_agent = create_tool_calling_agent(llm, [semantic_scholar_tool], semantic_prompt)
semantic_scholar_executor = AgentExecutor(agent=semantic_scholar_agent, tools=[semantic_scholar_tool], verbose=True)

wiki_tools = load_tools(["wikipedia"], llm=llm)
wiki_agent = initialize_agent(wiki_tools, llm, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

def gather_resources_for_topic(topic):
    collected_data = ""
    youtube_results = youtube_agent_chain.run(f"{topic}")
    collected_data += f"### YouTube Results:\n{youtube_results}\n\n"
    semantic_query = f"query: {topic}\nFor each relevant paper, create 3 sections: paper name with URL, summary, and citations."
    semantic_results = semantic_scholar_executor.invoke({"input": semantic_query})
    collected_data += f"### Semantic Scholar Results:\n{semantic_results}\n\n"
    wiki_results = wiki_agent.run(topic)
    collected_data += f"### Wikipedia Summary:\n{wiki_results}\n\n"
    return collected_data

def summarize_all_resources(all_text):
    final_prompt = (
        "You are an expert in summarizing and organizing research information. "
        "You will be given data collected from various sources like YouTube, Semantic Scholar and Wikipedia.\n\n"
        "Structure the following into a readable summary:\n\n"
        f"{all_text}\n\n"
        "Make it detailed, readable, and categorized into relevant sections."
    )
    return llm.invoke(final_prompt)

# ------------------------- STREAMLIT APP -------------------------
def app():
    st.title("End-to-End Research Assistant")
    st.subheader("Ideate, Expand, Collect and Summarize Research Effortlessly")

    topic = st.text_area("Enter your initial research idea or topic:")

    if st.button("Generate and Collect Insights"):
        if topic:
            with st.spinner("Generating ideas..."):
                ai_generated_text = generate_research_insights(topic)
                st.write("### AI-Generated Research Ideas")
                st.write(ai_generated_text)

            topics = extract_topics_from_text(ai_generated_text)
            if not topics:
                st.warning("Could not extract additional topics from the AI response.")
                return

            st.write("### Extracted Topics")
            for t in topics:
                st.markdown(f"- {t}")

            with st.spinner("Collecting and organizing research..."):
                combined_resources = ""
                for t in topics:
                    st.write(f"Processing: {t}")
                    combined_resources += gather_resources_for_topic(t)

                summarized_output = summarize_all_resources(combined_resources)

            st.write("### Final Summarized Research Report")
            st.write(summarized_output)
        else:
            st.warning("Please enter a topic to begin.")

if __name__ == "__main__":
    app()
