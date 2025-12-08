"""
Biomedical Research Analysis System
 - Collect research from PubMed/PMC 
 _/ Analyze and summarize research using OpenAI LLM
 - Integrate results
"""
import os
import re
import json
import random
import asyncio
from typing import Dict, List, Any
from datetime import datetime
from abc import ABC, abstractmethod
from openai import OpenAI, AsyncOpenAI
from Fetch_data import ResearchPaper
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Abstract Base Classes
# ============================================================================
class LLMAnalyzer(ABC):
    """Abstract base class for LLM-based paper analysis"""
    
    @abstractmethod
    def analyze(self, paper: ResearchPaper, theme: str) -> Dict[str, Any]:
        """Analyze if paper is relevant to given theme"""
        pass


# ============================================================================
# OpenAI LLM Analyzer Implementation
# ============================================================================

class OpenAIAnalyzer(LLMAnalyzer):
    """Analyzer using OpenAI's API for paper relevance assessment"""
    def __init__(self, api_key: str = None, model: str = "gpt-4o-mini", base_url: str = None):
        """
        Initialize OpenAI analyzer
        
        Args:
            api_key: OpenAI API key
            model: Model to use (default: gpt-4o-mini for cost efficiency)
        """
        self.model = model
        self.base_url = base_url
        self.kwargs = self._validate_api_key(api_key, base_url)
        self.semaphore = asyncio.Semaphore(5) # max_concurrent
        
    def _validate_api_key(self, api_key: str, base_url: str):
        """Validate OpenAI API key, and listing available models"""
        if not api_key: # get api_key from .env file if not provided
            api_key=os.environ["OPENAI_API_KEY"]
            custom_api_url=os.environ["OPENAI_API_BASE"] # optional custom API base URL
            if not api_key: # didn't find api_key in .env file
                logger.info("API key cannot be empty, please set your OpenAI API key as an environment variable OPENAI_API_KEY")
                return

        kwargs = {"api_key": api_key}
        if "custom_api_url" in locals() and custom_api_url:
            kwargs["base_url"] = custom_api_url
        if base_url:
            kwargs["base_url"] = base_url

        try:
            self.client = OpenAI(**kwargs)
            models = self.client.models.list() # fetch available models
            self._available_models = [m.id for m in models.data]
            logger.info(f"Available models: {self._available_models }") 
            logger.info("OPENAI_API_KEY is set and is valid")
        except Exception as e:
            logger.warning(f"Unexpected error: {str(e)}")

        return kwargs
    
    def generate_pubmed_query(self, query: str, advanced_options: dict) -> str:
        """
        Generate a ready-to-use PubMed query string for a given query from user.
        """
        query_en = self._translate_to_english(query)
        logger.info(f"Generating PubMed search paramters for query: {query}")

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content":  self._create_query_prompt(query_en, advanced_options)},
                    {"role": "user", "content": query_en}
                ],
                temperature=0.3
            )

            pubmed_query = response.choices[0].message.content.strip()
            match = re.search(r'(\([^\n]+)', pubmed_query, re.DOTALL)
            if match:
                pubmed_query = match.group(1).strip()
            logger.info(f"Generated PubMed search paramters: {pubmed_query}")
            return pubmed_query

        except Exception as e:
            raise RuntimeError(f"Error generating PubMed query: {e}")
        
    def _translate_to_english(self, text: str) -> str:
        """
        Optional: Translate Chinese to English for PubMed search
        """
        if all(ord(c) < 128 for c in text):  # simple check for non-Chinese text
            return text
        
        logger.info(f"Translating query from Chinese to English: {text}")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a professional biomedical translator."},
                    {"role": "user", "content": f"Translate this research topic into academic English: {text}"}
                ],
                temperature=0.2
            )
            translation = response.choices[0].message.content.strip()
            logger.info(f"Translated to: {translation}")
            return translation
        except Exception as e:
            logger.error(f"Translation failed: {e}", exc_info=True)
            return text  # fallback to original text

    def _create_query_prompt(self, query: str, advanced_options: dict) -> str:
        """
        Create LLM prompt to generate PubMed search query
        """
        start_year, end_year = advanced_options["year_range"]
        year_filter = f'("{start_year}"[Date - Publication] : "{end_year}"[Date - Publication])'
        types = advanced_options["article_types"]
        if types:
            type_filter = " OR ".join([f'"{t}"[Publication Type]' for t in types])
            type_filter = f"({type_filter})"
            parts = [year_filter, type_filter, "English[Language]", "hasabstract[text]"]

        else:
            parts = [year_filter, "English[Language]", "hasabstract[text]"]
        mandatory_filters = " AND ".join(parts)
        SYSTEM_PROMPT = (
        "You are an expert in biomedical literature retrieval and PubMed Advanced Search query construction."
        "REQUIREMENT (MUST be followed):"
        "1) Consider the user's query:"+query+" carefully and determine the most relevant search terms and synonyms. Do this internally — DO NOT output any chain-of-thought, analysis, or explanations. Unless you cannot construct a PubMed Advanced Search query based on the current user's query."
        "2) Output ONLY a single-line PubMed Advanced Search query that can be pasted directly into PubMed's query box. Do NOT output any JSON, lists, commentary, or additional text."
        "3) Build a concise Boolean query using AND / OR and parentheses. Use PubMed field tags:"
        " - Prefer [MeSH Terms] for official subject headings where appropriate."
        " - Use [Title/Abstract] for synonyms, abbreviations, drug names, or phrases."
        "4) Mandatory filters (these MUST appear in the final query exactly as shown):"+mandatory_filters+"."
        "5) Keep the query focused: avoid introducing broad synonyms that generate excessive noise."
        "6) Ensure parentheses are balanced and field tags are used correctly."
        "Respond **only** PubMed Advanced Search query without any other characters or punctuation or quotation marks, e.g. (Lipids[MeSH Terms] OR cholesterol[Title/Abstract]) AND English[Language]"
        )
        return SYSTEM_PROMPT

    async def async_analyze(self, papers: List[ResearchPaper], theme: str, analyze_bar, progress) -> List[ResearchPaper]:
        """
        Analyze if papers are relevant to the given theme concurrently
        
        Args:
            papers: ResearchPaper object papers searched from PubMed
            theme: Theme (user's query) to check relevance
            
        Returns:
            ResearchPaper object papers with analysis results
        """
        client = AsyncOpenAI(**self.kwargs)
        async with self.semaphore:
            logger.info(f"Analyzing papers for theme: {theme}")
            SYSTEM_PROMPT = (
                "You are an expert biomedical research analyst. Analyze the following set of research papers and evaluate both their thematic relevance to :\""+theme+"\" and the credibility of their publication sources."
                "Each paper includes: pmid(paper identifier); journal(the name of the publication journal); title(paper title); abstract(paper abstract text)."
                "Your tasks:"
                "1. Determine whether the paper abstract is *directly* relevant, *indirectly* relevant, or *irrelevant* to the theme, considering if the study addresses mechanisms, treatments, datasets, or risk factors related to the theme."
                "2. Evaluate if the journal is authoritative and influential in biomedical research, considering whether it is a well-known or society-backed journal (e.g., Nature, Cell, JAMA, The Lancet, Circulation, BMJ, NEJM, PLOS, BMC, etc.), classify the credibility as: *High*, *Medium*, or *Low*."
                "3. Write a 3 to 4 sentences summary to encapsulate the main findings of the paper abstract, as if explaining to a general educated audience."
                "Respond **only** with valid JSON using the structure below:"
                """
                {
                    "papers": [
                        {
                        "id": pmid(paper identifier),
                        "relevance_level": "direct" or "indirect" or "irrelevant",
                        "journal_credibility": "High" or "Medium" or "Low",
                        "simplified_summary": "3 to 4 sentences summary explaining the paper and encapsulating the main findings",
                        },
                        ...
                    ]
                }
                """
            )
            USER_PROMPT = (f"""Please start analyzing the following papers:
                            {json.dumps([paper.__dict__ for paper in papers], indent=2)}
                            """
                            )
            try:
                response = await client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": USER_PROMPT}
                    ],
                    temperature=0.3,
                    response_format={"type": "json_object"}
                )

                results = self._extract_analyses(response)
                progress[0] += 1
                analyze_bar.progress(progress[0] / progress[1]) # display the progress of the fetching on the streamlit bar

                return results
            
            except Exception as e:
                raise RuntimeError(f"Error analyzing paper relevance: {e}")

    def _extract_analyses(self, response):
        """
        Extract analysis results from LLM response, handling JSON parsing and formatting issues.
        """
        data = self._process_LLMresponse(response)
        papers = []
        if isinstance(data, dict):
            papers = next(iter(data.values()))
        elif isinstance(data, list):
            papers = data
        elif isinstance(data, str):
            papers.append(data)

        papers_analy = []
        for idx, result in enumerate(papers):
            if not isinstance(result, dict):
                print(f"⚠️ Skipping non-dict entry at index {idx}: {result}")
                continue
            else:
                # Create paper with analysis results
                paper = ResearchPaper(
                    pmid=result.get('id', ''),
                    relevance_level = result.get('relevance_level', 'irrelevant'),
                    journal_credibility = result.get('journal_credibility', 'Unknown'),
                    simplified_summary = result.get('simplified_summary', ''),
                    analysis_timestamp = datetime.now().isoformat()
                )
                papers_analy.append(paper)
        return papers_analy

    def analyze(self, paper: ResearchPaper, theme: str) -> ResearchPaper:
        """
        Analyze if paper is relevant to the given theme
        
        Args:
            paper: ResearchPaper object paper searched from PubMed
            theme: Theme (user's query) to check relevance
            
        Returns:
            ResearchPaper object paper with analysis results
        """
        logger.info(f"Analyzing papers for theme: {theme}")
        SYSTEM_PROMPT = (
            "You are an expert biomedical research analyst. Analyze the following research paper and evaluate both thematic relevance to :\""+theme+"\" and the credibility of publication sources."
            "Each paper includes: pmid(paper identifier); journal(the name of the publication journal); title(paper title); abstract(paper abstract text)."
            "Your tasks:"
            "1. Determine whether the paper is *directly*, *indirectly*, or *not* relevant to the theme, considering if the study addresses mechanisms, treatments, datasets, or risk factors related to the theme."
            "2. Evaluate if the journal is authoritative and influential in biomedical research, considering whether it is a well-known or society-backed journal (e.g., Nature, Cell, JAMA, The Lancet, Circulation, BMJ, NEJM, PLOS, BMC, etc.), whether it is indexed in MEDLINE or widely cited. Classify the credibility as: *High*, *Medium*, or *Low* and provide a short justification."
            "3. For relevant paper only, write a short, accessible 3 to 4 sentences summary as if explaining to a general educated audience."
            "Respond **only** with valid JSON using the structure below:"
            """
            {
                "papers": [
                    {
                    "id": pmid(paper identifier),
                    "relevance_level": "direct" or "indirect" or "irrelevant",
                    "journal_credibility": "High" or "Medium" or "Low",
                    "simplified_summary": "3 to 4 sentences summary explaining the paper (empty if not relevant)",
                    },
                    ...
                ]
            }
            """
        )
        USER_PROMPT = (f"""Please start analyzing the following paper:
                        {json.dumps(paper.__dict__, indent=2)}
                        """
                        )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_PROMPT}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )

            results = self._extract_analyses(response)
            return results
        
        except Exception as e:
            raise RuntimeError(f"Error analyzing paper relevance: {e}")
            
    def generate_report(self, user_query: str, contents: str) -> str:
        """
        Generate a summary report for the given theme and content
        Args:
            user_query: User's query
            contents: A single string concatenating LLM-generated short summaries (each separated by \n\n).
        Returns:
            Summary report
        """
        logger.info(f"Generating report...")
        SYSTEM_PROMPT = (
            "You are an expert research analyst who writes vivid, comprehensive, and reader-friendly scientific reports. "
            "Your task is to synthesize multiple research paper summaries into one clear and engaging final report. "
            "Guidelines: "
            "1. Maintain high factual accuracy based on the summaries — do not invent data or conclusions. "
            "2. When referencing any specific finding or claim, include its PMID in parentheses — e.g.: (PMID: 12345678). "
            "3. Integrate all summaries smoothly, highlighting key findings, contrasts, and trends. "
            "4. Explain the significance of the research relative to the user's query. "
            "5. Write in a structured and vivid narrative style suitable for an informed non-specialist reader. "
            "6. Avoid repetition; instead, synthesize and connect insights logically. "
            "7. Use natural, clear language with transitions that make the report flow smoothly. "
            "8. Return the final result in a JSON object with the following structure: "
            "{'summary_report': 'string'}"
        )
        USER_PROMPT = (
            f"User query: {user_query}\n\n"
            f"Analyzed paper summaries (each prefixed by its PMID, e.g.: pmid. summmary):\n{contents}\n\n"
            "Please generate a final integrated report that answers the user's query, "
            "fully incorporating and properly referencing the findings from all summaries. "
            "Include PMID references inline (e.g.: (PMID: 12345678)) when referring to each paper, "
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_PROMPT}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            report = self._process_LLMresponse(response, "summary_report")
            return report
        
        except Exception as e:
            raise RuntimeError(f"Error generating report: {e}")
        
    def integrate_reports(self, user_query: str, base_report: str, supplement: str) -> str:
        """Integrate multiple reports into one final report."""
        SYSTEM_PROMPT = (
            "You are an expert scientific writer."
            "Your task is to carefully integrate the essential and factual content from report2 into report1, "
            "enhancing and expanding report1 while ensuring logical completeness and scientific accuracy. "
            "Guidelines: "
            "1. The final report must be based on report1, with additional details incorporated from report2. "
            "2. Do NOT invent, reinterpret, or add any information that is not explicitly present in the two reports. "
            "3. Preserve all factual details, analyses, and citations (e.g., PMID references) exactly as they appear. "
            "4. Integrate without redundancy—merge overlapping points smoothly while keeping all key ideas intact. "
            "5. Ensure the final text reads as one logically structured, fluent, and cohesive narrative. "
            "6. Maintain scientific tone, clarity, and factual integrity throughout. "
            "7. Return the final result in a JSON object with the following structure: "
            "{'integrated_report': 'string'}"
        )
        USER_PROMPT = (
            f"The following are two scientific reports derived from the same user research query.\n\n"
            f"User query: {user_query}\n\n"
            f"Report1 (base report):\n{base_report}\n\n"
            f"Report2 (to integrate):\n{supplement}\n\n"
            "Integrate the key content from Report2 into Report1, generate a comprehensive and cohesive final report."
            "Do not add any new information beyond the two reports, and keep existing citations."
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_PROMPT}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            report = self._process_LLMresponse(response, "integrated_report")
            return report
        
        except Exception as e:
            raise RuntimeError(f"Error generating report: {e}")
        
    def _process_LLMresponse(self, response: str, key: str=None):
        """Process LLM response to extract final result."""
        if hasattr(response, "choices"):
            content = response.choices[0].message.content
        else:
            content = str(response)
            logger.warning(f"Failed to get choices from response: {content}")
            
        # JSON content may be wrapped in ```json ... ``` or ``` ... ```
        match = re.search(r"```(?:json)?(.*?)```", content, re.DOTALL | re.IGNORECASE)
        if match:
          content = match.group(1).strip()
        else:
            content = content.strip()
            if not content.startswith("{"):
                # &&&&&&&&&&&&& debug_LLM_response &&&&&&&&&&&&&
                file_name = 'test/match_fail'+str(random.randint(0, 100))+'.json'
                with open(file_name, "w", encoding="utf-8") as f:
                    f.write(content)
                # &&&&&&&&&&&&& debug_LLM_response &&&&&&&&&&&&&
                logger.warning(f"Failed to match ``` in response, see '{file_name}'.")

        # extract result from JSON content
        try:
            content_json = json.loads(content)
            if key:
                result = content_json.get(key, "")
            else:
                # logger.warning(f"No keywords for LLM response extraction, return raw JSON.")
                result = content_json
            return result
        except json.JSONDecodeError as e:
            # &&&&&&&&&&&&& debug_LLM_response &&&&&&&&&&&&&
            file_name = 'test/JSON_fail'+str(random.randint(0, 100))+'.json'
            with open(file_name, "w", encoding="utf-8") as f:
                f.write(content)
            # &&&&&&&&&&&&& debug_LLM_response &&&&&&&&&&&&&
            logger.error(f"Failed to parse JSON response, see '{file_name}'.", exc_info=True)
            raise e

