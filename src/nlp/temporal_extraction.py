"""
Temporal Information Extraction
Extracts dates, times, and temporal expressions from text
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dateutil import parser as date_parser
from dateutil.relativedelta import relativedelta

logger = logging.getLogger(__name__)


class TemporalExtractor:
    """
    Extract temporal information from text
    
    Features:
    - Absolute dates (2024-01-15, January 15 2024)
    - Relative dates (yesterday, last week, 3 days ago)
    - Time expressions (3:00 PM, morning, evening)
    - Durations (2 hours, 3 months)
    - Date ranges (from Jan to March, 2020-2024)
    """
    
    def __init__(self, reference_date: Optional[datetime] = None):
        """
        Initialize temporal extractor
        
        Args:
            reference_date: Reference date for relative expressions
        """
        self.reference_date = reference_date or datetime.now()
        
        # Compile regex patterns
        self._compile_patterns()
        
        logger.info("✓ TemporalExtractor initialized")
        logger.info(f"  Reference date: {self.reference_date.strftime('%Y-%m-%d')}")
    
    def _compile_patterns(self):
        """Compile regex patterns for temporal expressions"""
        
        # Absolute date patterns
        self.patterns = {
            # ISO format: 2024-01-15
            'iso_date': re.compile(r'\b(\d{4})-(\d{2})-(\d{2})\b'),
            
            # US format: 01/15/2024, 1/15/24
            'us_date': re.compile(r'\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b'),
            
            # Written: January 15, 2024 or Jan 15 2024
            'written_date': re.compile(
                r'\b(January|February|March|April|May|June|July|August|September|October|November|December|'
                r'Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)'
                r'\s+(\d{1,2}),?\s+(\d{4})\b',
                re.IGNORECASE
            ),
            
            # Just year: in 2024, during 2023
            'year_only': re.compile(r'\b(in|during|since|by)\s+(\d{4})\b', re.IGNORECASE),
            
            # Month and year: January 2024
            'month_year': re.compile(
                r'\b(January|February|March|April|May|June|July|August|September|October|November|December|'
                r'Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)'
                r'\s+(\d{4})\b',
                re.IGNORECASE
            ),
            
            # Relative dates
            'relative_day': re.compile(
                r'\b(yesterday|today|tomorrow)\b',
                re.IGNORECASE
            ),
            
            'relative_week': re.compile(
                r'\b(last|this|next)\s+(week|month|year|quarter)\b',
                re.IGNORECASE
            ),
            
            'days_ago': re.compile(
                r'\b(\d+)\s+(days?|weeks?|months?|years?)\s+ago\b',
                re.IGNORECASE
            ),
            
            # Time expressions
            'time_24h': re.compile(r'\b(\d{1,2}):(\d{2})\b'),
            
            'time_12h': re.compile(
                r'\b(\d{1,2}):(\d{2})\s*(AM|PM|am|pm)\b'
            ),
            
            # Durations
            'duration': re.compile(
                r'\b(\d+)\s+(seconds?|minutes?|hours?|days?|weeks?|months?|years?)\b',
                re.IGNORECASE
            ),
            
            # Date ranges
            'date_range': re.compile(
                r'\b(from|between)\s+(.+?)\s+(to|and|until)\s+(.+?)\b',
                re.IGNORECASE
            ),
        }
    
    def extract(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract all temporal expressions from text
        
        Args:
            text: Input text
        
        Returns:
            List of temporal expression dicts
        """
        temporal_entities = []
        
        # Extract different types
        temporal_entities.extend(self._extract_absolute_dates(text))
        temporal_entities.extend(self._extract_relative_dates(text))
        temporal_entities.extend(self._extract_times(text))
        temporal_entities.extend(self._extract_durations(text))
        temporal_entities.extend(self._extract_date_ranges(text))
        
        # Sort by position in text
        temporal_entities.sort(key=lambda x: x['start'])
        
        # Remove duplicates (same span)
        unique_entities = []
        seen_spans = set()
        
        for entity in temporal_entities:
            span = (entity['start'], entity['end'])
            if span not in seen_spans:
                unique_entities.append(entity)
                seen_spans.add(span)
        
        return unique_entities
    
    def _extract_absolute_dates(self, text: str) -> List[Dict[str, Any]]:
        """Extract absolute date expressions"""
        dates = []
        
        # ISO format
        for match in self.patterns['iso_date'].finditer(text):
            year, month, day = match.groups()
            try:
                date = datetime(int(year), int(month), int(day))
                dates.append({
                    'text': match.group(0),
                    'start': match.start(),
                    'end': match.end(),
                    'type': 'absolute_date',
                    'value': date.isoformat(),
                    'normalized': date.strftime('%Y-%m-%d')
                })
            except ValueError:
                continue
        
        # Written dates
        for match in self.patterns['written_date'].finditer(text):
            try:
                date = date_parser.parse(match.group(0))
                dates.append({
                    'text': match.group(0),
                    'start': match.start(),
                    'end': match.end(),
                    'type': 'absolute_date',
                    'value': date.isoformat(),
                    'normalized': date.strftime('%Y-%m-%d')
                })
            except:
                continue
        
        # Month and year
        for match in self.patterns['month_year'].finditer(text):
            try:
                date = date_parser.parse(match.group(0))
                dates.append({
                    'text': match.group(0),
                    'start': match.start(),
                    'end': match.end(),
                    'type': 'month_year',
                    'value': date.isoformat(),
                    'normalized': date.strftime('%Y-%m')
                })
            except:
                continue
        
        # Year only
        for match in self.patterns['year_only'].finditer(text):
            year = match.group(2)
            dates.append({
                'text': match.group(0),
                'start': match.start(),
                'end': match.end(),
                'type': 'year',
                'value': year,
                'normalized': year
            })
        
        return dates
    
    def _extract_relative_dates(self, text: str) -> List[Dict[str, Any]]:
        """Extract relative date expressions"""
        dates = []
        
        # Relative days
        for match in self.patterns['relative_day'].finditer(text):
            word = match.group(1).lower()
            
            if word == 'yesterday':
                date = self.reference_date - timedelta(days=1)
            elif word == 'today':
                date = self.reference_date
            elif word == 'tomorrow':
                date = self.reference_date + timedelta(days=1)
            
            dates.append({
                'text': match.group(0),
                'start': match.start(),
                'end': match.end(),
                'type': 'relative_date',
                'value': date.isoformat(),
                'normalized': date.strftime('%Y-%m-%d')
            })
        
        # Days ago
        for match in self.patterns['days_ago'].finditer(text):
            amount = int(match.group(1))
            unit = match.group(2).lower()
            
            if 'day' in unit:
                date = self.reference_date - timedelta(days=amount)
            elif 'week' in unit:
                date = self.reference_date - timedelta(weeks=amount)
            elif 'month' in unit:
                date = self.reference_date - relativedelta(months=amount)
            elif 'year' in unit:
                date = self.reference_date - relativedelta(years=amount)
            
            dates.append({
                'text': match.group(0),
                'start': match.start(),
                'end': match.end(),
                'type': 'relative_date',
                'value': date.isoformat(),
                'normalized': date.strftime('%Y-%m-%d')
            })
        
        # Last/this/next week/month/year
        for match in self.patterns['relative_week'].finditer(text):
            modifier = match.group(1).lower()
            unit = match.group(2).lower()
            
            if modifier == 'last':
                delta = -1
            elif modifier == 'this':
                delta = 0
            elif modifier == 'next':
                delta = 1
            
            if unit == 'week':
                date = self.reference_date + timedelta(weeks=delta)
            elif unit == 'month':
                date = self.reference_date + relativedelta(months=delta)
            elif unit == 'year':
                date = self.reference_date + relativedelta(years=delta)
            elif unit == 'quarter':
                date = self.reference_date + relativedelta(months=delta*3)
            
            dates.append({
                'text': match.group(0),
                'start': match.start(),
                'end': match.end(),
                'type': 'relative_date',
                'value': date.isoformat(),
                'normalized': date.strftime('%Y-%m-%d')
            })
        
        return dates
    
    def _extract_times(self, text: str) -> List[Dict[str, Any]]:
        """Extract time expressions"""
        times = []
        
        # 12-hour format
        for match in self.patterns['time_12h'].finditer(text):
            hour = int(match.group(1))
            minute = int(match.group(2))
            meridiem = match.group(3).upper()
            
            if meridiem == 'PM' and hour != 12:
                hour += 12
            elif meridiem == 'AM' and hour == 12:
                hour = 0
            
            times.append({
                'text': match.group(0),
                'start': match.start(),
                'end': match.end(),
                'type': 'time',
                'value': f"{hour:02d}:{minute:02d}",
                'normalized': f"{hour:02d}:{minute:02d}"
            })
        
        # 24-hour format
        for match in self.patterns['time_24h'].finditer(text):
            hour = int(match.group(1))
            minute = int(match.group(2))
            
            if 0 <= hour <= 23 and 0 <= minute <= 59:
                times.append({
                    'text': match.group(0),
                    'start': match.start(),
                    'end': match.end(),
                    'type': 'time',
                    'value': f"{hour:02d}:{minute:02d}",
                    'normalized': f"{hour:02d}:{minute:02d}"
                })
        
        return times
    
    def _extract_durations(self, text: str) -> List[Dict[str, Any]]:
        """Extract duration expressions"""
        durations = []
        
        for match in self.patterns['duration'].finditer(text):
            amount = int(match.group(1))
            unit = match.group(2).lower()
            
            # Normalize to seconds
            if 'second' in unit:
                seconds = amount
            elif 'minute' in unit:
                seconds = amount * 60
            elif 'hour' in unit:
                seconds = amount * 3600
            elif 'day' in unit:
                seconds = amount * 86400
            elif 'week' in unit:
                seconds = amount * 604800
            elif 'month' in unit:
                seconds = amount * 2592000  # Approximate
            elif 'year' in unit:
                seconds = amount * 31536000  # Approximate
            
            durations.append({
                'text': match.group(0),
                'start': match.start(),
                'end': match.end(),
                'type': 'duration',
                'value': seconds,
                'normalized': f"{amount} {unit}"
            })
        
        return durations
    
    def _extract_date_ranges(self, text: str) -> List[Dict[str, Any]]:
        """Extract date range expressions"""
        ranges = []
        
        for match in self.patterns['date_range'].finditer(text):
            start_text = match.group(2).strip()
            end_text = match.group(4).strip()
            
            try:
                start_date = date_parser.parse(start_text)
                end_date = date_parser.parse(end_text)
                
                ranges.append({
                    'text': match.group(0),
                    'start': match.start(),
                    'end': match.end(),
                    'type': 'date_range',
                    'value': {
                        'start': start_date.isoformat(),
                        'end': end_date.isoformat()
                    },
                    'normalized': f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
                })
            except:
                continue
        
        return ranges
    
    def extract_simple(self, text: str) -> List[str]:
        """
        Simple extraction returning just temporal strings
        
        Args:
            text: Input text
        
        Returns:
            List of temporal expression strings
        """
        entities = self.extract(text)
        return [entity['text'] for entity in entities]
    
    def normalize_date(self, date_text: str) -> Optional[str]:
        """
        Normalize date text to ISO format
        
        Args:
            date_text: Date in any format
        
        Returns:
            ISO format date string (YYYY-MM-DD) or None
        """
        try:
            date = date_parser.parse(date_text)
            return date.strftime('%Y-%m-%d')
        except:
            return None


# ==========================================
# TESTING
# ==========================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    extractor = TemporalExtractor()
    
    print("\n" + "="*80)
    print("TEMPORAL EXTRACTION TESTS")
    print("="*80)
    
    test_texts = [
        "India's GDP grew 8% in 2024 according to data released yesterday",
        "The meeting is scheduled for January 15, 2024 at 3:00 PM",
        "He worked there from 2020 to 2023 for 3 years",
        "The event happened last week on 2024-01-10",
        "The project will take 6 months starting next month",
        "Data from March 2024 shows significant growth",
        "The conference is tomorrow at 9:30 AM",
        "Sales increased 15% between January and March 2024",
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n{i}. Text: {text}")
        print("-" * 80)
        
        temporal_entities = extractor.extract(text)
        
        if temporal_entities:
            print("  Temporal expressions:")
            for entity in temporal_entities:
                print(f"    • {entity['text']} → {entity['type']}: {entity['normalized']}")
        else:
            print("  No temporal expressions found")
    
    print("\n" + "="*80)