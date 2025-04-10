import streamlit as st
from langchain_community.tools import YouTubeSearchTool
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_community.tools.semanticscholar.tool import SemanticScholarQueryRun
from langchain_community.utilities.semanticscholar import SemanticScholarAPIWrapper
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain_community.tools.playwright.utils import create_async_playwright_browser
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain_community.tools.pubmed.tool import PubmedQueryRun
from langchain.tools import tool
import asyncio
import nest_asyncio
from langchain_together import ChatTogether
import re

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop) 
nest_asyncio.apply()

# If you are using Playwright in a multi-threaded environment, 
# you should create a playwright instance per thread.
# here because we are using it inside of streamlit, which has its own thread/loop (An event loop runs in a thread )
# we have to set another loop that we will use to work with playwright browser and other agents
# so that we dont stall the streamlit UI
# this is another advanced techniques that will help you a lot


# Initialize the LLM
llm = ChatTogether(api_key=st.secrets['togetherai_apikey'], temperature=0.0, model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo")

# Initialize YouTube Search Tool
# youtube_tool = YouTubeSearchTool()

@tool
def youtube_transcript_tool(query: str) -> str:
    """Returns the transcript and YouTube URL of the top video for a given query."""
    try:
        tool = YouTubeSearchTool(language="en")
        search_results = tool.run(f"{query},5")  # likely returns a formatted string with video links
        video_ids = re.findall(r"v=([^&\s]+)", search_results)

        if not video_ids:
            return "No valid video IDs found in search result."

        total_text = ""
        for video_id in video_ids:
            video_link = f"https://www.youtube.com/watch?v={video_id}"
            try:
                transcript = YouTubeTranscriptApi.get_transcript(video_id)
                transcript_text = ' '.join([t['text'] for t in transcript])
                total_text += f"Transcript of {video_link}:\n{transcript_text}\n\n"
            except Exception as inner_e:
                total_text += f"Could not fetch transcript for {video_link}: {str(inner_e)}\n\n"

        return total_text
    except Exception as e:
        return f"Error during video search.\nReason: {str(e)}"
    
# Use your custom tool
yt_tools = [youtube_transcript_tool]

youtube_agent_chain = initialize_agent(
    yt_tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)

# Initialize Semantic Scholar Tool
semantic_scholar_api = SemanticScholarAPIWrapper(doc_content_chars_max=1000, top_k_results=5)
semantic_scholar_tool = SemanticScholarQueryRun(api_wrapper=semantic_scholar_api)



# async_browser = create_async_playwright_browser()
# toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=async_browser)
# playwright_tools = toolkit.get_tools()

# Create a tool-calling agent for Semantic Scholar
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert researcher."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

semantic_scholar_agent = create_tool_calling_agent(llm, [semantic_scholar_tool], prompt)
semantic_scholar_executor = AgentExecutor(agent=semantic_scholar_agent, tools=[semantic_scholar_tool], verbose=True)

# # Initialize Playwright-based agent
# playwright_agent = initialize_agent(
#     playwright_tools,
#     llm,
#     agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
#     verbose=True,
#     handle_parsing_errors=True
# )

wiki_tools = load_tools(["wikipedia"], llm=llm)
wiki_agent= initialize_agent(
    wiki_tools,
    llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose = True)

# Function to run all the tools, format the output, and structure it using LLM
async def collect_and_format_resources(topic):
    collected_data = ""

    # YouTube Search
    youtube_results = youtube_agent_chain.run(f"{topic}")
    st.write("got youtube results")
    collected_data += f"### YouTube Results:\n{youtube_results}\n\n"

    # Semantic Scholar Search
    semantic_query = f"query: {topic}\nFor each relevant paper, create 3 sections: paper name with URL, summary, and citations."
    semantic_scholar_response = semantic_scholar_executor.invoke({"input": semantic_query})
    st.write("got Semantic Scholar Results")
    collected_data += f"### Semantic Scholar Results:\n{semantic_scholar_response}\n\n"

    # # Playwright Web Search
    # web_search_query = f'find 5 articles/blogs on "{topic}" and return the top 5 article/blogs with the names and their URL for each. Do not use wikipedia blogs'
    # web_results = await playwright_agent.ainvoke(web_search_query)
    # st.write("got Web Articles and Blogs")
    # collected_data += f"### Web Articles and Blogs:\n{web_results}\n\n"

    # Wikipedia Search
    wiki_search_query = f'{topic}'
    wiki_results = wiki_agent(wiki_search_query)
    st.write("got Wiki Articles Summary")
    collected_data += f"### Wiki Articles Summary:\n{wiki_results}\n\n"

    # Final prompt to structure the data
    #         "3. **Web Articles and Blogs**: A list of the top articles and blogs with their URLs.\n\n"
    final_prompt = (
        "You are an expert in summarizing and organizing research information. "
        "You will be given data collected from various sources, such as YouTube videos, academic papers and wikipedia articles summary "
        "Please structure this data into a well-organized, readable report with the following sections:\n\n"
        "1. **YouTube Videos**: A list of the top videos related to the topic with brief descriptions.\n"
        "2. **Academic Papers (Semantic Scholar)**: A list of papers, their summaries, and key citations for each of the papers.\n"
        "3. **Wikipedia Articles**: A summary of the relevant Wikipedia articles.\n\n"
        "Here is the data to organize:\n\n"
        f"{collected_data}\n"
        "Please ensure the final output is well-structured and easy to understand."
    )

    # Get the final structured output from LLM
    structured_response = llm.generate([final_prompt]).generations[0][0].text

    return structured_response

# Streamlit UI
def app():
    st.title("Research Collection: Gather and Organize Resources")
    st.subheader("Effortlessly collect and organize research materials.")

    st.write("""
    Input a specific topic, question, or area of interest, and the AI will retrieve relevant academic papers, 
    articles, YouTube videos, and blogs. The assistant will organize these resources for easier review and synthesis.
    """)

    topic = st.text_area("Enter your research topic or question:")

    if st.button("Collect Resources"):
        if topic:
            with st.spinner('Collecting and organizing resources...'):
                structured_result = loop.run_until_complete(collect_and_format_resources(topic))
                st.write("### Structured Research Summary")
                st.write(structured_result)
        else:
            st.warning("Please enter a topic to collect resources.")

if __name__ == "__main__":
    app()
