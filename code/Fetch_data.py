"""
Biomedical Research Analysis System
 _/ Collect research from PubMed/PMC 
 - Analyze and summarize research using OpenAI LLM
 - Integrate results
"""
import os
import time
from itertools import cycle
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import requests
import asyncio
import aiohttp
import xml.etree.ElementTree as ET
import json
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class ResearchPaper:
    """Represents a research paper with metadata and analysis results"""
    pmid: str
    title: str = None
    authors: List[str] = None
    publication_date: str = None
    article_types: List[str] = None
    journal: Optional[str] = None
    pmc_cited: Optional[int] = None
    doi: Optional[str] = None
    pmc_id: Optional[str] = None
    pubmed_url: Optional[str] = None
    pmc_url: Optional[str] = None
    abstract: Dict[str, str] = None

    # Analysis results
    is_relevant: Optional[bool] = None
    relevance_reason: Optional[str] = None
    relevance_level: Optional[str] = None  # "direct", "indirect", "irrelevant"
    simplified_summary: Optional[str] = None
    journal_credibility: Optional[str] = None  # "High", "Medium", "Low"
    credibility_reason: Optional[str] = None
    analysis_timestamp: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return asdict(self)
    
    def get_urls(self) -> Dict[str, str]:
        """Get all available URLs for the paper"""
        urls = {}
        if self.pubmed_url:
            urls['pubmed'] = self.pubmed_url
        if self.pmc_url:
            urls['pmc'] = self.pmc_url
        if self.doi:
            urls['doi'] = f"https://doi.org/{self.doi}"
        return urls


# ============================================================================
# Abstract Base Classes
# ============================================================================

class BiomedicalDatabaseClient(ABC):
    """Abstract base class for biomedical database API clients"""
    
    @abstractmethod
    def search(self, query: str, max_results: int = 10) -> List[str]:
        """Search for papers and return list of IDs"""
        pass
    
    @abstractmethod
    def fetch_details(self, paper_ids: List[str]) -> List[ResearchPaper]:
        """Fetch detailed information for given paper IDs"""
        pass


# ============================================================================
# PubMed API Client Implementation
# ============================================================================

@dataclass
class PubMedSearchParams:
    """Search parameters for PubMed queries"""
    query: str
    max_results: int = 10
    
    # Date range filters
    date_from: Optional[str] = None  # Format: YYYY/MM/DD or YYYY/MM or YYYY
    date_to: Optional[str] = None    # Format: YYYY/MM/DD or YYYY/MM or YYYY
    date_type: str = "pdat"  # pdat (publication), edat (entrez), mdat (modification)
    
    # Publication type filters
    publication_types: Optional[List[str]] = None  # e.g., ["Clinical Trial", "Meta-Analysis"]
    
    # Article filters
    article_types: Optional[List[str]] = None  # e.g., ["Journal Article", "Review"]
    
    # Language filter
    languages: Optional[List[str]] = None  # e.g., ["eng", "fre"]
    
    # Species filter
    species: Optional[List[str]] = None  # e.g., ["humans", "mice"]
    
    # Journal filter
    journals: Optional[List[str]] = None  # e.g., ["Nature", "Science"]
    
    # Field-specific search
    title_keywords: Optional[List[str]] = None
    author_names: Optional[List[str]] = None
    mesh_terms: Optional[List[str]] = None  # Medical Subject Headings
    
    # Age groups
    age_groups: Optional[List[str]] = None  # e.g., ["Adult", "Child", "Aged"]
    
    # Sex filter
    sex: Optional[str] = None  # "male", "female", or None for both
    
    # Text availability
    has_abstract: bool = False
    free_full_text: bool = False
    
    # Sort options(default is relevance)
    sort_by: str = "relevance"  # relevance, pub_date, or other PubMed sort options
    
    def build_query(self) -> str:
        """Build PubMed query string with all filters"""
        query_parts = [self.query]
        
        # Add title keywords
        if self.title_keywords:
            title_query = " OR ".join([f'"{kw}"[Title]' for kw in self.title_keywords])
            query_parts.append(f"({title_query})")
        
        # Add author filters
        if self.author_names:
            author_query = " OR ".join([f'"{auth}"[Author]' for auth in self.author_names])
            query_parts.append(f"({author_query})")
        
        # Add MeSH terms
        if self.mesh_terms:
            mesh_query = " OR ".join([f'"{term}"[MeSH Terms]' for term in self.mesh_terms])
            query_parts.append(f"({mesh_query})")
        
        # Add publication types
        if self.publication_types:
            pub_type_query = " OR ".join([f'"{pt}"[Publication Type]' for pt in self.publication_types])
            query_parts.append(f"({pub_type_query})")
        
        # Add article types
        if self.article_types:
            art_type_query = " OR ".join([f'"{at}"[Filter]' for at in self.article_types])
            query_parts.append(f"({art_type_query})")
        
        # Add language filter
        if self.languages:
            lang_query = " OR ".join([f"{lang}[Language]" for lang in self.languages])
            query_parts.append(f"({lang_query})")
        
        # Add species filter
        if self.species:
            species_query = " OR ".join([f'"{sp}"[MeSH Terms]' for sp in self.species])
            query_parts.append(f"({species_query})")
        
        # Add journal filter
        if self.journals:
            journal_query = " OR ".join([f'"{j}"[Journal]' for j in self.journals])
            query_parts.append(f"({journal_query})")
        
        # Add age group filter
        if self.age_groups:
            age_query = " OR ".join([f'"{ag}"[MeSH Terms]' for ag in self.age_groups])
            query_parts.append(f"({age_query})")
        
        # Add sex filter
        if self.sex:
            query_parts.append(f'"{self.sex}"[MeSH Terms]')
        
        # Add text availability filters
        if self.has_abstract:
            query_parts.append("hasabstract")
        
        if self.free_full_text:
            query_parts.append("free full text[Filter]")
        
        # Combine all parts with AND
        return " AND ".join(query_parts)


class PubMedClient(BiomedicalDatabaseClient):
    """Client for interacting with PubMed/PMC APIs"""
    
    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    
    def __init__(self, api_key: Optional[str] = None, email: Optional[str] = None, api_accounts: Optional[List[Tuple[str, str]]] = None):
        """
        Initialize PubMed client
        
        Args:
            api_key: NCBI API key (recommended for higher rate limits)
            email: Email address (required by NCBI guidelines)
            api_accounts: List of tuples of (API key, email)
        """
        self.api_key = os.getenv('NCBI_API_KEY') if api_key is None else api_key
        self.email = os.getenv('NCBI_EMAIL') if email is None else email
        self.api_accounts = api_accounts if api_accounts is not None else list(zip(os.getenv("NCBI_API_KEYS", "").split(","), 
                                                                                os.getenv("NCBI_EMAILS", "").split(",")))
        self.account_cycle = cycle(self.api_accounts)
        self.session = requests.Session()
        
    def _make_request(self, endpoint: str, params: Dict[str, Any]) -> Dict:
        """Make API request with retry logic"""
        if self.api_key:
            params['api_key'] = self.api_key
        if self.email:
            params['email'] = self.email
        
        url = f"{self.BASE_URL}/{endpoint}"
        
        for attempt in range(3):
            try:
                response = self.session.get(url, params=params, timeout=30)
                logger.info(f"HTTP Response: {response}")
                response.raise_for_status()
                # time.sleep(0.34)  # Respect NCBI rate limits (3 requests/sec)
                return response.json()
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request attempt {attempt + 1} failed: {e}")
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
        
    def search(
        self, 
        query: Optional[str] = None, 
        search_params: Optional[PubMedSearchParams] = None,
        max_results: int = 10,
        year_range: Optional[Tuple[int, int]] = None,
        llm_search: bool = False
    ) -> List[str]:
        """
        Search PubMed for papers matching query
        
        Args:
            query: search query string
            search_params: Advanced search parameters object, PubMedSearchParams object with all search criteria
            max_results: Maximum number of results to return
            year_range: Tuple of (start_year, end_year) to filter by publication date
            llm_search: If True, use llm-generated search parameters(sort by relevance)

        Returns:
            List of PubMed IDs (PMIDs)
        """
        # Use search_params if provided, otherwise create basic params from query
        if llm_search:
            params = {
                'db': 'pubmed',
                'term': query,
                "retmode": "json",
                'retmax': max_results,
                'sort': "relevance"
            }
        else:
            if search_params:
                params_obj = search_params
            elif query:
                params_obj = PubMedSearchParams(query=query, max_results=max_results, date_from=year_range[0], date_to=year_range[1], date_type="pdat")
            else:
                raise ValueError("Either query or search_params must be provided")
            
            # Build the query string with all filters
            search_query = params_obj.build_query()
            logger.info(f"Searching PubMed: {search_query}")
            params = {
                'db': 'pubmed',
                'term': search_query,
                "retmode": "json",
                'retmax': params_obj.max_results,
                'sort': params_obj.sort_by
            }
            # Add date range if specified
            if params_obj.date_from or params_obj.date_to:
                if params_obj.date_from:
                    params['mindate'] = params_obj.date_from
                if params_obj.date_to:
                    params['maxdate'] = params_obj.date_to
                params['datetype'] = params_obj.date_type
        
        results = self._make_request('esearch.fcgi', params)
        pmids = results.get('esearchresult', {}).get('idlist', [])
        
        logger.info(f"Found {len(pmids)} papers")
        return pmids

    def advanced_search(self, search_params: PubMedSearchParams) -> List[str]:
        """
        Convenience method for advanced search with PubMedSearchParams
        
        Args:
            search_params: PubMedSearchParams object with all search criteria
            
        Returns:
            List of PubMed IDs (PMIDs)
        """
        return self.search(search_params=search_params)
    
    def fetch_brief(self, paper_ids: List[str]) -> List[ResearchPaper]:
        """
        Fetch brief information for given PMIDs, without abstract, faster
        
        Args:
            paper_ids: List of PubMed IDs
            
        Returns:
            Results(without abstract, Dict format) returned by PubMed API
        """
        batch_size = 200   
        results = []
        if not paper_ids:
            logger.info(f"No paper IDs provided")
            return []
        # Fetch in batches if too many
        elif len(paper_ids) > batch_size:
            logger.info(f"Fetching metadata for {len(paper_ids)} papers")
            logger.info(f"Paper count {len(paper_ids)} exceeds batch size {batch_size}, splitting into batches")

            for i in range(0, len(paper_ids), batch_size):
                batch_ids = paper_ids[i:i + batch_size]
                logger.info(f"Processing batch {i // batch_size + 1}: {len(batch_ids)} papers")
                params = {
                    'db': 'pubmed',
                    "retmode": "json",
                    'id': ','.join(batch_ids),
                }
                batch_result = self._make_request('esummary.fcgi', params)
                results.append(batch_result)
        else:
            logger.info(f"Fetching metadata for {len(paper_ids)} papers")
            params = {
                'db': 'pubmed',
                "retmode": "json",
                'id': ','.join(paper_ids),
            }
            results.append(self._make_request('esummary.fcgi', params))

        return results
    
    def fetch_details(self, paper_ids: List[str]) -> List[ResearchPaper]:
        """
        Fetch detailed information for given PMIDs, with abstract
        
        Args:
            paper_ids: List of PubMed IDs
            
        Returns:
            List of ResearchPaper objects
        """
        batch_size = 200
        papers = []
        if not paper_ids:
            logger.info(f"No paper IDs provided")
            return []
        # Fetch in batches if too many
        elif len(paper_ids) > batch_size:
            logger.info(f"Paper count {len(paper_ids)} exceeds batch size {batch_size}, splitting into batches")

            for i in range(0, len(paper_ids), batch_size):
                batch_ids = paper_ids[i:i + batch_size]
                logger.info(f"Processing batch {i // batch_size + 1}: {len(batch_ids)} papers")
                params = {
                    'db': 'pubmed',
                    "retmode": "json",
                    'id': ','.join(batch_ids),
                }
                batch_result = self._make_request('esummary.fcgi', params)
                papers.extend(self._get_fetched(batch_ids, batch_result, fetch_abstract=True))

                # Save every 5 batches
                if i % (5 * batch_size) == 0:  
                    self._export_papers_to_json(papers, output_file="raw_papers_details.json")
                time.sleep(1)
        else:
            params = {
                'db': 'pubmed',
                "retmode": "json",
                'id': ','.join(paper_ids),
            }
            result = self._make_request('esummary.fcgi', params)
            papers = self._get_fetched(paper_ids, result, fetch_abstract=True)
            self._export_papers_to_json(papers, output_file="raw_papers_details.json")

        return papers
    
    def _get_fetched(self, paper_ids: List[str], result: Dict[str, Any], 
                     fetch_abstract: bool=False) -> List[ResearchPaper]:
        """Helper to process fetched results"""
        papers = []
        for pmid in paper_ids:
            try:
                paper_data = result['result'].get(pmid)
                if not paper_data:
                    logger.warning(f"No data found for PMID {pmid}")
                    continue
                
                # Extract authors
                authors = [
                    author.get('name', '') 
                    for author in paper_data.get('authors', [])
                ]
                # Create ResearchPaper object
                paper = ResearchPaper(
                    pmid=pmid,
                    title=paper_data.get('title', ''),
                    authors=authors,
                    publication_date=paper_data.get('pubdate', ''),
                    article_types=paper_data.get('pubtype', ''),
                    journal=paper_data.get('fulljournalname', ''),
                    pmc_cited=paper_data.get('pmcrefcount', ''),
                    pubmed_url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                )
                if fetch_abstract == True:
                    paper.abstract = self._fetch_abstract(pmid)
                    paper.doi = self._extract_doi(paper_data)
                    paper.pmc_id = self._extract_pmc_id(paper_data)
                    # if paper.pmc_id:
                    #     paper.pmc_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{paper.pmc_id}/"
                papers.append(paper)
                
            except Exception as e:
                logger.error(f"Error processing paper {pmid}: {e}")
                continue
        return papers

    def _fetch_abstract(self, pmid: str) -> Optional[Dict[str, str]]:
        """
        Fetch abstract for given PMID(only one paper at a time)
        
        Args:
            pmid: PubMed ID for the paper
            
        Returns:
            Dictionary with abstract sections or None if not found
        """
        logger.info(f"Fetching metadata and abstract for {pmid}")
        params = {
            'db': 'pubmed',
            'id': pmid,
            'rettype': 'abstract',
            'retmode': 'xml'
        }
        url = f"{self.BASE_URL}/efetch.fcgi"
        
        if self.api_key:
            params['api_key'] = self.api_key
        if self.email:
            params['email'] = self.email

        for attempt in range(3):
            try:
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()

                # processing XML
                root = ET.fromstring(response.text)
                abstract_elements = root.findall(".//Abstract/AbstractText")

                if not abstract_elements:
                    logger.warning(f"No abstract found for PMID {pmid}")
                    return None

                # Check and label abstract(structured or unstructured)
                if any("Label" in el.attrib for el in abstract_elements):
                    abstract_dict = {
                        el.attrib.get("Label", "UNLABELED"): (el.text or "").strip()
                        for el in abstract_elements
                    }
                else:
                    abstract_dict = {
                        "Unstructured": " ".join(
                            (el.text or "").strip() for el in abstract_elements if el.text
                        )
                    }
                return abstract_dict

            except requests.exceptions.RequestException as e:
                logger.warning(f"Request attempt {attempt + 1} failed: {e}")
                if attempt == 2:
                    raise
                time.sleep(attempt)
    
    async def _fetch_one(self, session: aiohttp.ClientSession, pmid, semaphore, extract_bar, progress) -> Optional[Dict[str, str]]:
        """
        Fetch abstract for given PMID(only one paper at a time) concurrently
        """
        params = {
            'db': 'pubmed',
            'id': pmid,
            'rettype': 'abstract',
            'retmode': 'xml'
        }
        url = f"{self.BASE_URL}/efetch.fcgi"
        api_key, email = next(self.account_cycle)
        params['api_key'] = api_key
        params['email'] = email
        if not self.api_key or not self.email:
            logger.error("PubMed API key or email is not set.")
        for attempt in range(3):
            try:
                async with semaphore:
                    async with session.get(url, params=params, timeout=30) as resp:
                        if resp.status != 200:
                            raise aiohttp.ClientError(f"HTTP {resp.status}")
                        text = await resp.text()
                        progress[0] += 1
                        extract_bar.progress(progress[0] / progress[1]) # display the progress of the fetching on the streamlit bar
                        logger.info(f"Fetched {pmid}")

                        return self._parse_xml(text, pmid)
            except Exception as e:
                logger.warning(f"[{pmid}] attempt {attempt+1} failed: {e}")
                await asyncio.sleep(1)

    def _parse_xml(self, xml_text: str, pmid: str) -> Optional[Dict[str, str]]:
        """Parse PubMed XML into structured/unstructured abstract"""
        try:
            root = ET.fromstring(xml_text)
            abstract_elements = root.findall(".//Abstract/AbstractText")
            if not abstract_elements:
                logger.warning(f"No abstract found for PMID {pmid}")
                return None

            # Check and label abstract(structured or unstructured)
            if any("Label" in el.attrib for el in abstract_elements):
                return {
                    el.attrib.get("Label", "UNLABELED"): (el.text or "").strip()
                    for el in abstract_elements
                }
            else:
                return {
                    "Unstructured": " ".join(
                        (el.text or "").strip() for el in abstract_elements if el.text
                    )
                }
        except ET.ParseError as e:
            logger.warning(f"XML parse failed for PMID {pmid}: {e}")
            return None
        
    def _extract_doi(self, paper_data: Dict) -> Optional[str]:
        """Extract DOI from paper data"""
        article_ids = paper_data.get('articleids', [])
        for id_obj in article_ids:
            if id_obj.get('idtype') == 'doi':
                return id_obj.get('value')
        return None
    
    def _extract_pmc_id(self, paper_data: Dict) -> Optional[str]:
        """Extract PMC ID from paper data"""
        article_ids = paper_data.get('articleids', [])
        for id_obj in article_ids:
            if id_obj.get('idtype') == 'pmc':
                return id_obj.get('value')
        return None

    def _export_papers_to_json(self, papers: List[ResearchPaper], output_file: str = "raw_papers.json"):
        """Export searched raw papers to structured JSON file"""
        output_folder = "data"
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, output_file)

        metadata = {
            "Description": "Raw PubMed papers data exported from PubMedClient",
            "total_papers": len(papers),
        }   
        data = []
        for paper in papers:
            paper_dict = {
                "title": paper.title,
                "authors": paper.authors,
                "publication_date": paper.publication_date,
                "article_types": paper.article_types,
                "journal": paper.journal,
                "pmc_cited": paper.pmc_cited,
                "doi": paper.doi if hasattr(paper, "doi") else None,
                "pmc_id": paper.pmc_id if hasattr(paper, "pmc_id") else None,
                "pmid": paper.pmid,
                "pubmed_url": paper.pubmed_url if hasattr(paper, "pubmed_url") else None,
                "abstract": paper.abstract if hasattr(paper, "abstract") else None,
            }
            data.append(paper_dict)

        output = {
            "metadata": metadata,
            "papers": data
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=4, ensure_ascii=False)

        logger.info(f"{len(data)} Papers exported to {output_path}")

