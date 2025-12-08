import streamlit as st
import plotly.express as px
from PIL import Image
import pandas as pd
import numpy as np
from Orchestrator import ResearchAnalyzer
from Fetch_data import PubMedClient
from LLM import OpenAIAnalyzer
import os
import json
import re
import asyncio


# Page configuration
st.set_page_config(
    page_title="Smart Analyst",
    page_icon="🧑‍⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Adjust width of sidebar */
    [data-testid="stSidebar"] {
            min-width: 280px;
            max-width: 320px;
        }
    /* Set “Press Enter to apply” invisible */
    .stTextInput input + div {
        display: none;
    }
    .my-slider [data-testid="stSlider"] {
        color: #334f39; !important;
    }
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
</style>
""", unsafe_allow_html=True)


# Big size data loading with caching
@st.cache_data
def load_data():
    """Load JSON data and convert to DataFrame"""
    with open("./data/Analysis.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    papers = data.get("papers", [])
    df = pd.DataFrame(papers)

    # Preprocess and clean data
    df["pmc_cited"] = pd.to_numeric(df.get("pmc_cited", 0), errors="coerce").fillna(0).astype(int)
    df["publication_year"] = df["publication_date"].astype(str).str.extract(r"(\d{4})").astype(int)
    
    return df

# Small size and frequent data updates, processing without caching
def preprocess_papers(paper_infos):
    for paper_info in paper_infos:
        paper_info = paper_info.get("result", {})
        if "uids" in paper_info:
            del paper_info["uids"]
        try:
            brief_papers.update(paper_info)
        except NameError:
            brief_papers = paper_info

    records = []
    for uid, info in brief_papers.items():
        pubmed_url=f"https://pubmed.ncbi.nlm.nih.gov/{uid}/"
        record = {
            "pmid": uid,
            "Title": info.get("title", "").strip(),
            "Journal": info.get("fulljournalname", info.get("source", "")),
            "Cited_by_in_PMC": int(info.get("pmcrefcount") or 0),
            "Pub_time": info.get("pubdate", ""),
            "Article_type": ", ".join(info.get("pubtype", [])),
            "Authors": ", ".join(a.get("name", "") for a in info.get("authors", [])),
            "PubMed_URL": pubmed_url
        }
        records.append(record)
    return records


def main():
    # Title 
    st.markdown('<h1 class="main-header">🧑‍⚕️ Smart Analyst</h1>', unsafe_allow_html=True)
    
    # ###################################################
    #                     Sidebar 
    # ###################################################
    with st.sidebar:
        st.header("Search Settings")

        # input your search query 
        query = st.text_area("🔍 Input Your Research Query", height=76)
        num_papers = st.slider("📚 Number of Papers", 10, 500, 200)
        year_range = st.slider("📅 Publication Year Range", 2010, 2025, (2020, 2025))
        LLMaugSearch = st.checkbox("LLM-Augmented Search", key=f"LLM augmented search")
        # advanced options
        with st.expander("Advanced Options"):
            min_citations = st.number_input("Minimum Citations", 0, 100, 0)
            selected_categories = st.multiselect(
                "Article Types",
                ["Clinical Trial", "Meta-Analysis", "Case Reports", "Review", "Systematic Review"],
                # default=["Clinical Trial", "Review", "Systematic Review"]
            )
    
        # click this button to run llm search
        left, right = st.columns([1.5, 3])
        with left:
            st.write("")
        with right:
            run = st.button("START")

        # show some example queries for each scenario
        example_queries = {
            "Medical Evidence-based Study":
                "💡 The effect of coffee on blood lipids  \n"  
                "💡 The impact of exercise on mental health  \n"
                "💡 The Application of Artificial Intelligence in Medical Diagnosis"
        }
        st.info(f"Recommended queries:  \n{example_queries['Medical Evidence-based Study']}")

        API_key = st.text_area("🔑 Input Your API key", height=40)
        Model = st.selectbox("🧠 Choose Model", ["gpt-5", "gpt-5-mini", "gpt-4.1-mini", "gpt-4o-mini", "gpt-3.5-turbo"], index=3)
        LLMor = st.checkbox("OpenRouter", key=f"OpenRouter")



    # ###################################################
    #                     Main Page
    # ###################################################
    if query and run:
        if not API_key:
            st.error("❌ Please input your API key")
            return
        elif LLMor:
            Analyst = ResearchAnalyzer(database_client=PubMedClient(), 
                                    llm_analyzer=OpenAIAnalyzer(api_key=API_key, model=Model, base_url="https://openrouter.ai/api/v1/") )
        elif API_key == "local":
            Analyst = ResearchAnalyzer(database_client=PubMedClient(), 
                                    llm_analyzer=OpenAIAnalyzer(api_key=None))
        elif API_key!="local" and Model: # OpenAI official API
            Analyst = ResearchAnalyzer(database_client=PubMedClient(), 
                                    llm_analyzer=OpenAIAnalyzer(api_key=API_key, model=Model) )
            
        # Status Bar: showing the workflow progress
        with st.status("👇 Smart Analyst is working and updating workflow...", expanded=True) as status:
            # placeholder
            col1, col2 = st.columns([1, 2])
            with col1:
                step1 = st.empty()
                step1.write(f"Step 1: Searching for your query...")
                step2 = st.empty()
                step2.write(f"Step 2: Extracting abstracts...")
                step3 = st.empty()
                step3.write(f"Step 3: Analyzing abstracts...") 
                step4 = st.empty()
                step4.write(f"Step 4: Generating summary report...")
            with col2:
                generate_bar = st.progress(0)
                extract_bar = st.progress(0)
                tip = st.empty()
                analyze_bar = st.progress(0)
                summary_bar = st.progress(0)

        # step 1: Search according to user's query   
        generate_bar.progress(1 / 4)
        if LLMaugSearch:
            status.update(label="🧠 Step 1/4: LLM is generating PubMed search parameters for your query...", state="running")
        else:
            status.update(label="🔎 Step 1/4: Searching PubMed for your query...", state="running")
        results = Analyst.search_papers(LLMaugSearch, user_query=query, max_results=num_papers,
                                        advanced_options={"year_range": year_range,
                                                          "min_citations": min_citations,
                                                          "article_types": selected_categories})
        generate_bar.progress(3 / 4)
        if len(results) != 3:
            status.update(label="❌ ERROR.", state="error")
            step1.write(f"⚠️ No papers found for the given query and filters. Please try adjusting your search criteria.")
            st.info(f"LLM responds: {results}")
        else:
            paper_ids, paper_infos, llm_query = results
            step1.write(f"✅ Step 1: Found {len(paper_ids)} papers.")
        generate_bar.progress(4 / 4)
        if LLMaugSearch:
            st.write(f"LLM generated PubMed query:")
            st.info(f"{llm_query}")
        papers_df = pd.DataFrame(preprocess_papers(paper_infos))

        # step 2: Fetch abstract content for papers found in Step 1 
        status.update(label="🦾 Step 2/4: Extracting papers' abstract...", state="running")
        paper_abstracts = asyncio.run(Analyst.async_extract_abstracts(paper_ids, extract_bar, max_concurrent=10))

        # merge abstracts content back to papers_df
        paper_abstracts_df = pd.DataFrame([paper.__dict__ for paper in paper_abstracts])
        pmid_to_abstract = dict(zip(paper_abstracts_df['pmid'].astype(str), paper_abstracts_df['abstract']))
        papers_df['Abstract'] = papers_df['pmid'].astype(str).map(pmid_to_abstract)
        before = len(papers_df)
        papers_df.dropna(inplace=True)
        step2.write(f"✅ Step 2: Got {len(paper_abstracts)} abstracts, {before - len(papers_df)} papers dropped due to missing values.")
        tip.markdown("""
        <style>
        .space {margin-top: 45px;}
        </style>
        <div class="space"></div>
        """,
        unsafe_allow_html=True)

        # step 3: LLM Read and analyze papers' abstract content...
        status.update(label="📉 Step 3/4: Analyzing and summarizing abstracts...", state="running")
        paper_analyzed = asyncio.run(Analyst.async_analyze_papers(paper_abstracts, query, analyze_bar, batch_size=10))
        step3.write(f"✅ Step 3: Finished analysis for {len(paper_analyzed)} papers.")
        # merge analysis results back to papers_df
        paper_analyzed_df = pd.DataFrame([paper.__dict__ for paper in paper_analyzed])     
        pmid_to_relevance_level = dict(zip(paper_analyzed_df['pmid'].astype(str), paper_analyzed_df['relevance_level']))
        papers_df['Relevance_level'] = papers_df['pmid'].astype(str).map(pmid_to_relevance_level)
        pmid_to_journal_credibility = dict(zip(paper_analyzed_df['pmid'].astype(str), paper_analyzed_df['journal_credibility']))
        papers_df['Journal_credibility'] = papers_df['pmid'].astype(str).map(pmid_to_journal_credibility)
        pmid_to_simplified_summary = dict(zip(paper_analyzed_df['pmid'].astype(str), paper_analyzed_df['simplified_summary']))
        papers_df['Simplified_summary'] = papers_df['pmid'].astype(str).map(pmid_to_simplified_summary)

        # ############# Data Processing ############# 
        # papers_df = pd.read_csv('2025-10-15T14-14_export.csv')
        papers_df["Pub_year"] = papers_df["Pub_time"].astype(str).str.extract(r"(\d{4})").astype(int)
        direct = papers_df[papers_df["Relevance_level"] == 'direct']
        indirect = papers_df[papers_df["Relevance_level"] == 'indirect']

        # normalise papers_df["Journal_credibility"]
        valid_levels = ["High", "Medium", "Low"]
        grouped = papers_df.groupby("Journal")["Journal_credibility"]
        journal_majority_map = {}
        for journal, values in grouped:
            valid_values = [v for v in values if v in valid_levels]
            if valid_values: 
                most_common = pd.Series(valid_values).mode().iloc[0] # "Journal_credibility"=majority vote
            else:
                most_common = "Low"
            journal_majority_map[journal] = most_common
        # replace valid_levels to majority "Journal_credibility" in the same Journal
        papers_df["Journal_credibility"] = [
            journal_majority_map[j] if c not in valid_levels else c
            for j, c in zip(papers_df["Journal"], papers_df["Journal_credibility"])]
        # replace all invalid values to "Low"
        papers_df["Journal_credibility"] = papers_df["Journal_credibility"].where(papers_df["Journal_credibility"].isin(valid_levels), "Low")
        # ############# Data Processing ############# 



        tab1, tab2, tab3 = st.tabs(["📊 Visualization", "👍 Relevant Papers", "🤖 LLM Analysis"])
        # ###################################################
        #             Page Tab 1 : Visualization
        # ###################################################
        with tab1:
            # ############ chart 1 - Papers overview #############
            # define bubble size mapping by Journal_credibility
            size_map = {"High": 1500, "Medium": 500, "Low": 100}
            papers_df["Journal_credibility"] = papers_df["Journal_credibility"].map(size_map)
            # process value for x-axis and y-axis
            papers_df["Published Year"] = papers_df["Pub_year"] + np.random.uniform(-0.2, 0.2, size=len(papers_df))
            papers_df['Cited'] = papers_df['Cited_by_in_PMC']+100
            fig = px.scatter(papers_df,
                            x="Published Year",
                            y="Cited",
                            size="Journal_credibility",
                            color="Relevance_level",
                            color_discrete_map={"direct": "#5A8CE8",
                                                "indirect": "#D5E4FF",
                                                "irrelevant": "#BFC0C4"}, 
                            category_orders={"Relevance_level": ["irrelevant", "indirect", "direct"]},
                            hover_name="Title",
                            custom_data=["Journal_credibility", "Relevance_level", "Cited_by_in_PMC", "Pub_year"],
                            height=600)
            # custom hover information
            fig.update_traces(hovertemplate=("<b>%{hovertext}</b><br>"   
                                            "Journal credibility: %{customdata[0]}<br>"  
                                            "Relevance level: %{customdata[1]}<br>"
                                            "Cited: %{customdata[2]}<br>" 
                                            "Year: %{customdata[3]}<br>"),
                            marker=dict(line=dict(width=1, color="#ffffff")))
            # hide irrelevant papers
            fig.for_each_trace(lambda trace: trace.update(visible="legendonly") if trace.name == "irrelevant" else trace.update(visible=True))
            fig.update_layout(template="plotly_white", legend_title_text="Relevance Level")
            # log y-axis
            raw_ticks = [0, 10, 100, 1000, 10000]
            shifted = [t + 100 for t in raw_ticks]   
            fig.update_yaxes(tickvals=shifted, ticktext=raw_ticks, type="log", title="Cited (log scale)")
            # set title and annotation
            fig.update_layout(title=dict(text="Papers Overview<br><sup> * Bubble size indicates journal credibility</sup>"))
            st.plotly_chart(fig)

            # ############ chart 2 - Group Relevance distribution ############ 
            group_size = 10
            papers_df = papers_df.reset_index(drop=True)
            papers_df["Group"] = (papers_df.index // group_size) + 1 
            group_counts = (
                papers_df.groupby(["Group", "Relevance_level"]).size().reset_index(name="Count")
            )
            color_map = {"direct": "#5A8CE8", "indirect": "#D5E4FF", "irrelevant": "#BFC0C4"}
            fig = px.bar(group_counts,
                        x="Group",
                        y="Count",
                        color="Relevance_level",
                        color_discrete_map=color_map,
                        barmode="group", 
                        title="Relevance Level Distribution",
                        category_orders={"Relevance_level": ["direct", "indirect", "irrelevant"]},
                        height=600)
            fig.update_layout(template="plotly_white",
                            xaxis_title="Group ("+str(group_size)+" papers each)",
                            yaxis_title="Count",
                            legend_title="Relevance Level")
            st.plotly_chart(fig)

            # ############ chart 3 - Direct relevant Journal distribution(top 10 journals by count) ############ 
            Journal_top10 = direct['Journal'].value_counts().head(10)
            fig_pie = px.pie(values=Journal_top10.values,
                            names=Journal_top10.index,
                            title="Direct Relevant Journal Distribution (top 10 journals by count)",
                            color_discrete_sequence=px.colors.qualitative.Pastel)
            fig_pie.update_traces(textfont_size=14, textposition='outside')
            st.plotly_chart(fig_pie)

            # ############ table 1 - Results in Dataframe  #############
            st.write("#### Papers Table")
            st.dataframe(papers_df[["Title", "Relevance_level", "Cited_by_in_PMC", "Pub_time", "Simplified_summary", "Journal", "Journal_credibility"]])
            csv = papers_df.to_csv(index=False)
            st.download_button(
                label=f"💾 Download as CSV",
                data=csv,
                file_name=f"Analysis_results.csv",
                mime="text/csv"
            )


        # ###################################################
        #         Page Tab 2 : LLM Analysis Result
        # ###################################################
        with tab2:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📄 Paper amounts", len(papers_df))
            with col2:
                st.metric("🥇 Directly Relevant", len(papers_df[papers_df['Relevance_level'] == 'direct']))
            with col3:
                st.metric("🥈 Indirectly Relevant", len(papers_df[papers_df['Relevance_level'] == 'indirect']))
                        
            default_expand_n = 3  # default expand first n papers
            default_expand_n = min(default_expand_n, len(direct), len(indirect))

            # ################ direct relevant papers ################
            st.markdown('<h2 class="main-header">Directly Relevant Papers</h2>', unsafe_allow_html=True)
            # show default expand first n papers
            for i, row in direct[:default_expand_n].iterrows():
                c1, c2 = st.columns([2, 1])
                with c1:
                    st.info(str(row.get("Simplified_summary", "")).strip())
                with c2:
                    st.write(f"**Title:** {row['Title']}")
                    st.write(f"**Journal:** {row['Journal']}")
                    st.write(f"**Relevance:** {row['Relevance_level']}")
                    st.markdown(f"**PubMed URL:** [Link]({row['PubMed_URL']})")
                    st.write("---")
            # show expander for the rest of papers
            with st.expander(f"👈 Expand All Directly Relevant Papers", expanded=False):
                st.caption(f"Showing last {len(direct)-default_expand_n} papers in category 'direct'")
                for i, row in direct[default_expand_n:].iterrows():
                    c1, c2 = st.columns([2, 1])
                    with c1:
                        st.info(str(row.get("Simplified_summary", "")).strip())
                    with c2:
                        st.write(f"**Title:** {row['Title']}")
                        st.write(f"**Journal:** {row['Journal']}")
                        st.write(f"**Relevance:** {row['Relevance_level']}")
                        st.markdown(f"**PubMed URL:** [Link]({row['PubMed_URL']})")
                        st.write("---")
            st.divider()
            # ################ indirect relevant papers ################
            st.markdown('<h2 class="main-header">Indirectly Relevant Papers</h2>', unsafe_allow_html=True)
            # show default expand first n papers
            for i, row in indirect[:default_expand_n].iterrows():
                c1, c2 = st.columns([5, 3])
                with c1:
                    st.info(str(row.get("Simplified_summary", "")).strip())
                with c2:
                    st.write(f"**Title:** {row['Title']}")
                    st.write(f"**Journal:** {row['Journal']}")
                    st.write(f"**Relevance:** {row['Relevance_level']}")
                    st.markdown(f"**PubMed URL:** [Link]({row['PubMed_URL']})")
                    st.write("---")
            # show expander for the rest of papers
            with st.expander(f"👈 Expand All Indirectly Relevant Papers", expanded=False):
                st.caption(f"Showing last {len(indirect)-default_expand_n} papers in category 'indirect'")
                for i, row in indirect[default_expand_n:].iterrows():
                    c1, c2 = st.columns([5, 3])
                    with c1:
                        st.info(str(row.get("Simplified_summary", "")).strip())
                    with c2:
                        st.write(f"**Title:** {row['Title']}")
                        st.write(f"**Journal:** {row['Journal']}")
                        st.write(f"**Relevance:** {row['Relevance_level']}")
                        st.markdown(f"**PubMed URL:** [Link]({row['PubMed_URL']})")
                        st.write("---")


        # ###################################################
        #        Page Tab 3 : LLM-generated report
        # ###################################################
        with tab3:
            # step 4: summary report
            status.update(label="📝 Step 4/4: Writing final report for your query...", state="running")
            summary_bar.progress(1 / 4)
            summaries = [f"{pmid}. {summary}" for pmid, summary in zip(direct['pmid'], direct['Simplified_summary'])]
            report = Analyst.report(query=query, contents=summaries)
            summary_bar.progress(3 / 4)

            # add link to PMID
            final_report = re.sub(
                r'PMID(?:s)?:\s*([^\)\n]+)',
                lambda m: "(PMID: " + "; ".join(
                    [f"[{pid}](https://pubmed.ncbi.nlm.nih.gov/{pid})"
                    for pid in re.findall(r"\d+", m.group(1))]
                ) + ")", report)

            st.markdown(final_report)
            summary_bar.progress(4 / 4)
            step4.write(f"✅ Step 4: Finished summary report.")
            status.update(label="✅ Successfully completed all tasks!", state="complete", expanded=False)
            st.download_button(
                label="Download Analysis Report",
                data=report,
                file_name=f"{query} --analysis_report.md",
                mime="text/markdown"
            )
              


    else:
        # initial welcome page
        # ---------- Section 1: Workflow ----------
        current_dir = os.path.dirname(__file__) 
        image_path = os.path.join(current_dir, "Workflow.png")
        image = Image.open(image_path)
        st.image(image, caption="Smart Analyst Workflow", width="stretch")

        # ---------- Section 2: User Scenarios (Two Columns) ----------
        st.markdown("#### 👥 User Scenarios")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("##### 🎓 Literature Research")
            st.markdown("""
<style>
.scroll-container {
    display: flex;
    overflow-x: auto;
    padding: 1rem 0;
    gap: 1rem;
    scroll-behavior: smooth;
}
.card {
    flex: 0 0 480px;  
    background-color: #f8f9fa;
    border-radius: 12px;
    padding: 1rem;
    box-shadow: 0 2px 6px rgba(0,0,0,0.1);
    transition: all 0.3s ease;
}
.card:hover {
    transform: translateY(-3px);
    box-shadow: 0 4px 10px rgba(0,0,0,0.15);
}
blockquote {
    color: #555;
    font-style: italic;
    margin: 0.5em 0;
    padding-left: 1em;
    border-left: 3px solid #ccc;
}
.scroll-wrapper {
    overflow-x: visible;
}
</style>

<div class="scroll-wrapper">
<div class="scroll-container">
    <div class="card">
        <blockquote>
        "Feeling lost in a sea of academic papers?"<br>
        "Need to write a literature review but dread the manual search?"
        </blockquote>
    </div>

        Gain a bird's-eye view of your field in minutes.
        Build your draft faster with our intelligent analysis report.
</div>
</div>
""", unsafe_allow_html=True)

        with col2:
            st.markdown("##### 👨‍⚕️ Evidence-Based Research")
            st.markdown("""
<style>
.scroll-container2 {
    display: flex;
    overflow-x: auto;
    padding: 1rem 0;
    gap: 1rem;
    scroll-behavior: smooth;
}
.card2 {
    flex: 0 0 480px;
    background-color: #f8f9fa;
    border-radius: 12px;
    padding: 1rem;
    box-shadow: 0 2px 6px rgba(0,0,0,0.1);
    transition: all 0.3s ease;
}
.card2:hover {
    transform: translateY(-3px);
    box-shadow: 0 4px 10px rgba(0,0,0,0.15);
}
blockquote {
    color: #555;
    font-style: italic;
    margin: 0.5em 0;
    padding-left: 1em;
    border-left: 3px solid #ccc;
}
.scroll-wrapper2 {
    overflow-x: hidden;
}
</style>

<div class="scroll-wrapper2">
<div class="scroll-container2">
    <div class="card2">
        <blockquote>
        "Overwhelmed by conflicting studies and online health claims?"<br>
        "Need trustworthy data to guide clinical decisions?"
        </blockquote>
    </div>

        Stay informed, critical, and confident in your evidence-based practice.
        Get concise, unbiased insights with links to the original sources.
</div>
</div>
""", unsafe_allow_html=True)
        
        # ---------- Section 3: Start ----------
        st.markdown("""
        #### 🚀 Start
        Enter your research query on the left, start the **intelligent paper analysis**!
        """)






if __name__ == "__main__":
    main()

