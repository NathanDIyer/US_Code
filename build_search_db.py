#!/usr/bin/env python3
"""
Build USC Search Database

This script creates a SQLite database for high-performance text search
by extracting content from the existing JSON index and USC text files.

Run this script after building the JSON index with build_usc_index.py
"""

import os
import json
import re
import sqlite3
from datetime import datetime
import sys

def load_usc_index():
    """Load the existing JSON index"""
    try:
        with open('usc_sections_index.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print("Error: usc_sections_index.json not found!")
        print("Please run 'python build_usc_index.py' first to create the JSON index.")
        return None
    except Exception as e:
        print(f"Error loading JSON index: {e}")
        return None

def extract_section_content(file_path, file_position, content_length):
    """Extract the actual content of a section from the text file"""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
            file.seek(file_position)
            content = file.read(content_length)
            return content.strip()
    except Exception as e:
        # Try with different encoding if UTF-8 fails
        try:
            with open(file_path, 'r', encoding='latin-1') as file:
                file.seek(file_position)
                content = file.read(content_length)
                return content.strip()
        except Exception as e2:
            print(f"Error reading section content from {file_path}: {e}")
            return ""

def clean_content_for_search(content):
    """Clean and normalize content for better search results"""
    if not content:
        return ""
    
    # Remove excessive whitespace
    content = re.sub(r'\s+', ' ', content)
    
    # Remove some special formatting that might interfere with search
    content = re.sub(r'\[.*?\]', ' ', content)  # Remove bracketed references
    content = re.sub(r'\(Release Point.*?\)', ' ', content)  # Remove release points
    
    return content.strip()

def build_search_database():
    """Build the SQLite search database from the JSON index"""
    print("Building USC SQLite Search Database...")
    print("=" * 50)
    
    # Load the JSON index
    index = load_usc_index()
    if not index:
        return False
    
    # Initialize SQLite database
    db_path = 'usc_search.db'
    
    # Remove existing database if it exists
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"Removed existing database: {db_path}")
    
    # Create new database
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()
    
    # Create tables
    cursor.execute('''
        CREATE TABLE sections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title_num TEXT NOT NULL,
            section_num TEXT NOT NULL,
            section_title TEXT NOT NULL,
            content TEXT NOT NULL,
            filename TEXT NOT NULL,
            file_position INTEGER NOT NULL,
            content_length INTEGER NOT NULL,
            subsections TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create indexes
    cursor.execute('CREATE INDEX idx_title_num ON sections(title_num)')
    cursor.execute('CREATE INDEX idx_section_num ON sections(section_num)')
    cursor.execute('CREATE INDEX idx_filename ON sections(filename)')
    
    # Create full-text search virtual table
    cursor.execute('''
        CREATE VIRTUAL TABLE sections_fts USING fts5(
            section_title, 
            content,
            content_id UNINDEXED
        )
    ''')
    
    # Create metadata table
    cursor.execute('''
        CREATE TABLE search_metadata (
            key TEXT PRIMARY KEY,
            value TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    print(f"Processing {index['metadata']['total_sections']} sections from {index['metadata']['total_files']} files...")
    
    total_sections = 0
    processed_files = 0
    
    # Process each file
    for filename, file_data in index['files'].items():
        file_path = os.path.join('txt', filename)
        
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
            continue
        
        print(f"Processing {filename}...")
        title_num = file_data['title_info']['title_num']
        file_sections = 0
        
        # Process each section in the file
        for section_num, section_data in file_data['sections'].items():
            try:
                # Extract the actual content
                content = extract_section_content(
                    file_path,
                    section_data['file_position'],
                    section_data['content_length']
                )
                
                # Clean content for better search
                cleaned_content = clean_content_for_search(content)
                
                if not cleaned_content:
                    print(f"Warning: Empty content for section {section_num} in {filename}")
                    continue
                
                # Convert subsections to JSON string
                subsections_json = json.dumps(section_data.get('subsections', []))
                
                # Insert into main table
                cursor.execute('''
                    INSERT INTO sections 
                    (title_num, section_num, section_title, content, filename, 
                     file_position, content_length, subsections)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    title_num,
                    section_num,
                    section_data['title'],
                    cleaned_content,
                    filename,
                    section_data['file_position'],
                    section_data['content_length'],
                    subsections_json
                ))
                
                # Get the inserted row ID
                section_id = cursor.lastrowid
                
                # Insert into FTS table
                cursor.execute('''
                    INSERT INTO sections_fts (section_title, content, content_id)
                    VALUES (?, ?, ?)
                ''', (
                    section_data['title'],
                    cleaned_content,
                    section_id
                ))
                
                file_sections += 1
                total_sections += 1
                
                # Commit every 100 sections for progress
                if total_sections % 100 == 0:
                    connection.commit()
                    print(f"  Progress: {total_sections} sections processed...")
                    
            except Exception as e:
                print(f"Error processing section {section_num} in {filename}: {e}")
                continue
        
        processed_files += 1
        print(f"  ✓ {filename}: {file_sections} sections added")
    
    # Final commit
    connection.commit()
    
    # Add metadata
    cursor.execute('''
        INSERT INTO search_metadata (key, value)
        VALUES ('build_date', ?), ('source_index_sections', ?), ('processed_sections', ?)
    ''', (
        datetime.now().isoformat(),
        str(index['metadata']['total_sections']),
        str(total_sections)
    ))
    
    connection.commit()
    connection.close()
    
    # Get database file size
    db_size = os.path.getsize(db_path) / (1024 * 1024)  # MB
    
    print("=" * 50)
    print(f"SQLite search database build complete!")
    print(f"Database file: {db_path}")
    print(f"Database size: {db_size:.2f} MB")
    print(f"Processed files: {processed_files}/{index['metadata']['total_files']}")
    print(f"Total sections: {total_sections}")
    print(f"Success rate: {(total_sections/index['metadata']['total_sections']*100):.1f}%")
    
    return True

def test_search_database():
    """Test the newly created search database"""
    print("\nTesting search database...")
    
    try:
        import search_database
        
        # Test basic functionality
        db = search_database.get_search_database()
        stats = db.get_database_stats()
        
        print(f"✓ Database loaded successfully")
        print(f"  Sections in database: {stats['section_count']}")
        print(f"  Titles covered: {stats['title_count']}")
        print(f"  Database size: {stats['database_size_mb']} MB")
        
        # Test a simple search
        results = db.simple_search("highway", max_results=5)
        print(f"✓ Test search 'highway' found {len(results)} results")
        
        if results:
            print(f"  First result: Title {results[0]['title_num']}, Section {results[0]['section_num']}")
        
        return True
        
    except Exception as e:
        print(f"Error testing database: {e}")
        return False

if __name__ == "__main__":
    print("USC SQLite Search Database Builder")
    print("This will create a high-performance search database from your existing JSON index.")
    print()
    
    if not os.path.exists('usc_sections_index.json'):
        print("Error: usc_sections_index.json not found!")
        print("Please run 'python build_usc_index.py' first.")
        sys.exit(1)
    
    # Build the database
    success = build_search_database()
    
    if success:
        # Test the database
        test_success = test_search_database()
        
        if test_success:
            print("\n🚀 Ready to use! Your search will now be lightning fast!")
            print("   Start your dashboard with 'python usc_dash_app.py'")
        else:
            print("\n⚠️  Database built but testing failed. Check for errors above.")
    else:
        print("\n❌ Database build failed. Check for errors above.")
        sys.exit(1) 