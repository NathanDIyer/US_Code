#!/usr/bin/env python3
"""
USC SQLite Search Database

This module provides high-performance text search capabilities for the USC dashboard.
It works alongside the existing JSON index to provide:
- Fast navigation (existing JSON index)
- Lightning-fast content search (this SQLite database)
- Advanced search syntax (phrases, OR, NOT, wildcards, proximity)
"""

import sqlite3
import os
import re
from typing import List, Dict, Any, Optional, Tuple
import json


# =============================================================================
# Advanced Query Parser
# =============================================================================

class AdvancedQueryParser:
    """
    Parse user-friendly search syntax into SQLite FTS5 queries.
    
    Supported syntax:
        - Phrases: "federal agency" (exact phrase)
        - OR: highway OR interstate (either term)
        - NOT: criminal -misdemeanor or criminal NOT tax (exclusion)
        - Wildcards: regulat* (prefix matching)
        - Proximity: environmental NEAR protection (within 10 words)
        - NEAR/N: word1 NEAR/5 word2 (within N words)
        - Title filter: title:18 (filter by USC title number)
        
    Default behavior:
        - Multiple terms without operators = AND (all required)
        - Case-insensitive operators
    """
    
    def __init__(self):
        # Regex patterns for parsing
        self.phrase_pattern = re.compile(r'"([^"]+)"')
        self.title_filter_pattern = re.compile(r'\btitle:(\d+[A-Z]?)\b', re.IGNORECASE)
        self.near_with_distance_pattern = re.compile(r'(\S+)\s+NEAR/(\d+)\s+(\S+)', re.IGNORECASE)
        self.near_pattern = re.compile(r'(\S+)\s+NEAR\s+(\S+)', re.IGNORECASE)
        self.not_pattern = re.compile(r'\bNOT\s+(\S+)', re.IGNORECASE)
        self.minus_pattern = re.compile(r'\s-(\S+)')
        self.or_pattern = re.compile(r'\bOR\b', re.IGNORECASE)
        self.wildcard_pattern = re.compile(r'(\w+)\*')
    
    def parse(self, query: str) -> Dict[str, Any]:
        """
        Parse a user query into components for FTS5 search.
        
        Args:
            query: User's search query string
            
        Returns:
            Dict with:
                - fts_query: The FTS5-compatible query string
                - title_filter: Extracted title number filter (or None)
                - search_terms: List of terms for highlighting
                - original_query: The original query
                - parsed_explanation: Human-readable explanation of parsing
        """
        if not query or not query.strip():
            return {
                'fts_query': '',
                'title_filter': None,
                'search_terms': [],
                'original_query': query,
                'parsed_explanation': 'Empty query'
            }
        
        original = query.strip()
        working = original
        explanations = []
        highlight_terms = []
        
        # Step 1: Extract title filter (remove from query)
        title_filter = None
        title_match = self.title_filter_pattern.search(working)
        if title_match:
            title_filter = title_match.group(1)
            working = self.title_filter_pattern.sub('', working).strip()
            explanations.append(f"Filter: Title {title_filter}")
        
        # Step 2: Preserve and extract phrases
        phrases = self.phrase_pattern.findall(working)
        phrase_placeholders = {}
        for i, phrase in enumerate(phrases):
            placeholder = f"__PHRASE_{i}__"
            phrase_placeholders[placeholder] = f'"{phrase}"'
            working = working.replace(f'"{phrase}"', placeholder, 1)
            highlight_terms.extend(phrase.split())
            explanations.append(f'Phrase: "{phrase}"')
        
        # Step 3: Handle NEAR/N proximity (must come before simple NEAR)
        near_n_matches = list(self.near_with_distance_pattern.finditer(working))
        for match in reversed(near_n_matches):  # Reverse to preserve positions
            word1, distance, word2 = match.groups()
            fts_near = f'NEAR({word1} {word2}, {distance})'
            working = working[:match.start()] + fts_near + working[match.end():]
            highlight_terms.extend([word1, word2])
            explanations.append(f"Proximity: {word1} within {distance} words of {word2}")
        
        # Step 4: Handle simple NEAR (default 10 words)
        near_matches = list(self.near_pattern.finditer(working))
        for match in reversed(near_matches):
            word1, word2 = match.groups()
            # Skip if this is part of a NEAR() function already
            if 'NEAR(' not in working[max(0,match.start()-5):match.start()]:
                fts_near = f'NEAR({word1} {word2}, 10)'
                working = working[:match.start()] + fts_near + working[match.end():]
                highlight_terms.extend([word1, word2])
                explanations.append(f"Proximity: {word1} within 10 words of {word2}")
        
        # Step 5: Handle NOT keyword
        not_matches = list(self.not_pattern.finditer(working))
        for match in reversed(not_matches):
            term = match.group(1)
            working = working[:match.start()] + f'NOT {term}' + working[match.end():]
            explanations.append(f"Exclude: {term}")
        
        # Step 6: Handle -term exclusion (convert to NOT)
        minus_matches = list(self.minus_pattern.finditer(working))
        for match in reversed(minus_matches):
            term = match.group(1)
            working = working[:match.start()] + f' NOT {term}' + working[match.end():]
            explanations.append(f"Exclude: {term}")
        
        # Step 7: Handle wildcards (FTS5 supports prefix*)
        wildcard_matches = list(self.wildcard_pattern.finditer(working))
        for match in wildcard_matches:
            term = match.group(1)
            highlight_terms.append(term)
            explanations.append(f"Wildcard: {term}*")
        
        # Step 8: Handle OR (FTS5 uses OR directly)
        if self.or_pattern.search(working):
            # Normalize OR to uppercase
            working = self.or_pattern.sub(' OR ', working)
            explanations.append("OR: matching either term")
        
        # Step 9: Restore phrase placeholders
        for placeholder, phrase in phrase_placeholders.items():
            working = working.replace(placeholder, phrase)
        
        # Step 10: Extract remaining simple terms for highlighting
        # Remove FTS5 operators and extract words
        simple_terms = re.findall(r'\b([a-zA-Z0-9]+)\b', 
                                   re.sub(r'\bNOT\b|\bOR\b|\bAND\b|\bNEAR\b', '', working, flags=re.IGNORECASE))
        for term in simple_terms:
            if term not in highlight_terms and not term.startswith('__'):
                highlight_terms.append(term)
        
        # Step 11: Clean up whitespace
        working = ' '.join(working.split())
        
        # Build explanation
        if not explanations:
            explanations.append(f"Search for all terms: {working}")
        
        return {
            'fts_query': working,
            'title_filter': title_filter,
            'search_terms': list(set(highlight_terms)),  # Deduplicate
            'original_query': original,
            'parsed_explanation': ' | '.join(explanations)
        }
    
    def get_syntax_help(self) -> str:
        """Return help text for search syntax"""
        return """
Advanced Search Syntax:
━━━━━━━━━━━━━━━━━━━━━━
"phrase"        Exact phrase match          "federal agency"
OR              Either term                  highway OR interstate  
-word           Exclude term                 criminal -misdemeanor
NOT word        Exclude term                 criminal NOT tax
word*           Wildcard prefix              regulat* → regulate, regulation
NEAR            Within 10 words              environmental NEAR protection
NEAR/5          Within N words               tax NEAR/3 penalty
title:18        Filter by USC title          title:18 fraud

Default: Multiple words require ALL terms (AND logic)
Example: tax fraud title:26 → finds "tax" AND "fraud" in Title 26
        """

class USCSearchDatabase:
    def __init__(self, db_path='usc_search.db'):
        self.db_path = db_path
        self.connection = None
        self.init_database()
    
    def init_database(self):
        """Initialize the SQLite database with required tables and indexes"""
        self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row  # Enable column access by name
        
        cursor = self.connection.cursor()
        
        # Create main sections table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title_num TEXT NOT NULL,
                section_num TEXT NOT NULL,
                section_title TEXT NOT NULL,
                content TEXT NOT NULL,
                filename TEXT NOT NULL,
                file_position INTEGER NOT NULL,
                content_length INTEGER NOT NULL,
                subsections TEXT,  -- JSON array of subsection titles
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create indexes for fast searching
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_title_num ON sections(title_num)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_section_num ON sections(section_num)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_filename ON sections(filename)')
        
        # Create full-text search virtual table
        cursor.execute('''
            CREATE VIRTUAL TABLE IF NOT EXISTS sections_fts USING fts5(
                section_title, 
                content,
                content_id UNINDEXED
            )
        ''')
        
        # Create metadata table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS search_metadata (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        self.connection.commit()
        print("✓ SQLite search database initialized")
    
    def get_section_count(self) -> int:
        """Get total number of sections in the database"""
        cursor = self.connection.cursor()
        cursor.execute("SELECT COUNT(*) FROM sections")
        return cursor.fetchone()[0]
    
    def search_sections(self, search_terms: List[str], title_filter: Optional[str] = None, max_results: int = 20000) -> List[Dict[str, Any]]:
        """
        Fast search for sections containing the specified terms
        
        Args:
            search_terms: List of terms to search for
            title_filter: Optional title number to filter by (e.g., "23")
            max_results: Maximum number of results to return
            
        Returns:
            List of section dictionaries matching the existing format
        """
        if not search_terms:
            return []
        
        cursor = self.connection.cursor()
        results = []
        
        try:
            # Build FTS query - search in both title and content
            search_query = ' AND '.join(f'"{term}"' for term in search_terms)
            
            # Base SQL query
            sql = '''
                SELECT s.title_num, s.section_num, s.section_title, s.content, 
                       s.filename, s.subsections
                FROM sections s
                JOIN sections_fts fts ON s.id = fts.content_id
                WHERE sections_fts MATCH ?
            '''
            
            params = [search_query]
            
            # Add title filter if specified
            if title_filter:
                sql += ' AND s.title_num = ?'
                params.append(title_filter)
            
            # Add limit
            sql += ' ORDER BY rank LIMIT ?'
            params.append(max_results)
            
            cursor.execute(sql, params)
            rows = cursor.fetchall()
            
            # Convert to the format expected by the dashboard
            for row in rows:
                # Parse subsections JSON
                subsections = []
                if row['subsections']:
                    try:
                        subsections = json.loads(row['subsections'])
                    except:
                        subsections = []
                
                # Highlight search terms in content (like the original search)
                content = row['content']
                highlight_pattern = '|'.join(re.escape(str(term)) for term in search_terms)
                highlighted_context = re.sub(
                    f'({highlight_pattern})', 
                    r'**\1**', 
                    content, 
                    flags=re.IGNORECASE
                )
                
                # Count occurrences
                content_lower = content.lower()
                occurrence_count = sum(content_lower.count(str(term).lower()) for term in search_terms)
                
                # Create result in the same format as existing search
                result = {
                    'title_num': row['title_num'],
                    'title_label': f"Title {row['title_num']}",
                    'section_num': row['section_num'],
                    'section_title': row['section_title'],
                    'context': highlighted_context,  # Use 'context' to match original format
                    'content': content,  # Also include raw content
                    'filename': row['filename'],
                    'subsections': subsections,
                    'occurrence_count': occurrence_count,
                    'search_terms': search_terms  # For highlighting
                }
                results.append(result)
            
            print(f"✓ SQLite search found {len(results)} results for: {search_terms}")
            return results
            
        except Exception as e:
            print(f"Error in SQLite search: {e}")
            return []
    
    def simple_search(self, search_text: str, title_filter: Optional[str] = None, max_results: int = 20000) -> List[Dict[str, Any]]:
        """
        Simple text search - splits search_text into terms and searches
        
        Args:
            search_text: Text to search for (will be split into terms)
            title_filter: Optional title number to filter by
            max_results: Maximum number of results to return
        """
        if not search_text or not search_text.strip():
            return []
        
        # Split search text into terms (simple word splitting)
        terms = [term.strip().lower() for term in search_text.split() if term.strip()]
        return self.search_sections(terms, title_filter, max_results)
    
    def advanced_search(self, query: str, title_filter: Optional[str] = None, max_results: int = 20000) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Advanced search with full query syntax support.
        
        Supports:
            - Phrases: "federal agency"
            - OR: highway OR interstate
            - NOT: criminal -misdemeanor
            - Wildcards: regulat*
            - Proximity: environmental NEAR protection
            - Title filter: title:18
        
        Args:
            query: User's search query with advanced syntax
            title_filter: Optional title override (query title: takes precedence)
            max_results: Maximum number of results to return
            
        Returns:
            Tuple of (results_list, parse_info_dict)
        """
        if not query or not query.strip():
            return [], {'fts_query': '', 'title_filter': None, 'search_terms': [], 
                       'original_query': query, 'parsed_explanation': 'Empty query'}
        
        # Parse the query
        parser = AdvancedQueryParser()
        parse_info = parser.parse(query)
        
        # Query title filter overrides parameter
        effective_title_filter = parse_info['title_filter'] or title_filter
        
        cursor = self.connection.cursor()
        results = []
        
        try:
            fts_query = parse_info['fts_query']
            
            if not fts_query.strip():
                return [], parse_info
            
            # Base SQL query
            sql = '''
                SELECT s.title_num, s.section_num, s.section_title, s.content, 
                       s.filename, s.subsections
                FROM sections s
                JOIN sections_fts fts ON s.id = fts.content_id
                WHERE sections_fts MATCH ?
            '''
            
            params = [fts_query]
            
            # Add title filter if specified
            if effective_title_filter:
                sql += ' AND s.title_num = ?'
                params.append(effective_title_filter)
            
            # Add limit and order by relevance
            sql += ' ORDER BY rank LIMIT ?'
            params.append(max_results)
            
            cursor.execute(sql, params)
            rows = cursor.fetchall()
            
            # Get search terms for highlighting
            search_terms = parse_info['search_terms']
            
            # Convert to the format expected by the dashboard
            for row in rows:
                # Parse subsections JSON
                subsections = []
                if row['subsections']:
                    try:
                        subsections = json.loads(row['subsections'])
                    except:
                        subsections = []
                
                # Highlight search terms in content
                content = row['content']
                if search_terms:
                    highlight_pattern = '|'.join(re.escape(str(term)) for term in search_terms)
                    highlighted_context = re.sub(
                        f'({highlight_pattern})', 
                        r'**\1**', 
                        content, 
                        flags=re.IGNORECASE
                    )
                else:
                    highlighted_context = content
                
                # Count occurrences
                content_lower = content.lower()
                occurrence_count = sum(content_lower.count(str(term).lower()) for term in search_terms) if search_terms else 0
                
                # Create result in the same format as existing search
                result = {
                    'title_num': row['title_num'],
                    'title_label': f"Title {row['title_num']}",
                    'section_num': row['section_num'],
                    'section_title': row['section_title'],
                    'context': highlighted_context,
                    'content': content,
                    'filename': row['filename'],
                    'subsections': subsections,
                    'occurrence_count': occurrence_count,
                    'search_terms': search_terms
                }
                results.append(result)
            
            print(f"✓ Advanced search found {len(results)} results for: {fts_query}")
            print(f"  Parsed: {parse_info['parsed_explanation']}")
            return results, parse_info
            
        except Exception as e:
            print(f"Error in advanced search: {e}")
            print(f"  FTS query was: {parse_info['fts_query']}")
            # On error, return empty results but include parse info for debugging
            parse_info['error'] = str(e)
            return [], parse_info
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get statistics about the search database"""
        cursor = self.connection.cursor()
        
        # Get section count
        cursor.execute("SELECT COUNT(*) FROM sections")
        section_count = cursor.fetchone()[0]
        
        # Get title count
        cursor.execute("SELECT COUNT(DISTINCT title_num) FROM sections")
        title_count = cursor.fetchone()[0]
        
        # Get database file size
        db_size = 0
        if os.path.exists(self.db_path):
            db_size = os.path.getsize(self.db_path)
        
        return {
            'section_count': section_count,
            'title_count': title_count,
            'database_size_mb': round(db_size / (1024 * 1024), 2),
            'database_path': self.db_path
        }
    
    def close(self):
        """Close the database connection"""
        if self.connection:
            self.connection.close()
            self.connection = None

# Global database instance
_search_db = None

def get_search_database() -> USCSearchDatabase:
    """Get or create the global search database instance"""
    global _search_db
    if _search_db is None:
        _search_db = USCSearchDatabase()
    return _search_db

def search_sections(search_terms: List[str], title_filter: Optional[str] = None, max_results: int = 10000) -> List[Dict[str, Any]]:
    """Convenience function for searching sections"""
    db = get_search_database()
    return db.search_sections(search_terms, title_filter, max_results)

def simple_search(search_text: str, title_filter: Optional[str] = None, max_results: int = 10000) -> List[Dict[str, Any]]:
    """Convenience function for simple text search"""
    db = get_search_database()
    return db.simple_search(search_text, title_filter, max_results)

def advanced_search(query: str, title_filter: Optional[str] = None, max_results: int = 10000) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Convenience function for advanced search with full syntax support.
    
    Returns:
        Tuple of (results_list, parse_info_dict)
    """
    db = get_search_database()
    return db.advanced_search(query, title_filter, max_results)

def get_search_syntax_help() -> str:
    """Get help text for advanced search syntax"""
    parser = AdvancedQueryParser()
    return parser.get_syntax_help()

def get_database_stats() -> Dict[str, Any]:
    """Convenience function for getting database statistics"""
    db = get_search_database()
    return db.get_database_stats() 