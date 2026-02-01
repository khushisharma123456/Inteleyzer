"""
Case Matching Service - Pharmacovigilance Duplicate Case Detection
Uses similarity algorithms to find matching cases before adding new ones
"""

from difflib import SequenceMatcher
from datetime import datetime, timedelta
import math

class CaseMatchingEngine:
    """
    Matches new adverse event cases with existing cases using:
    - Drug name matching
    - Symptom similarity (fuzzy string matching)
    - Patient demographics (age, gender)
    - Time-based proximity (recent reports)
    """
    
    def __init__(self, threshold=0.75):
        """
        Initialize the matching engine
        
        Args:
            threshold (float): Similarity score threshold (0-1). 
                             Default 0.75 means 75% match needed
        """
        self.threshold = threshold
    
    def calculate_text_similarity(self, text1, text2):
        """
        Calculate similarity between two strings using SequenceMatcher
        
        Args:
            text1, text2 (str): Texts to compare
            
        Returns:
            float: Similarity score (0-1)
        """
        if not text1 or not text2:
            return 0.0
        
        # Normalize: lowercase, strip whitespace
        text1 = str(text1).lower().strip()
        text2 = str(text2).lower().strip()
        
        # Use SequenceMatcher for fuzzy matching
        matcher = SequenceMatcher(None, text1, text2)
        return matcher.ratio()
    
    def calculate_age_similarity(self, age1, age2, max_age_diff=10):
        """
        Calculate similarity based on age proximity
        
        Args:
            age1, age2 (int): Patient ages
            max_age_diff (int): Maximum age difference considered (default 10 years)
            
        Returns:
            float: Similarity score (0-1)
        """
        if not age1 or not age2:
            return 1.0  # No penalty if age missing
        
        age_diff = abs(int(age1) - int(age2))
        
        if age_diff <= max_age_diff:
            # Scale down similarity based on age difference
            return 1.0 - (age_diff / (max_age_diff * 2))
        
        return 0.5  # Low score if ages very different
    
    def calculate_case_similarity(self, new_case, existing_case):
        """
        Calculate overall similarity between new and existing case
        
        Scoring breakdown:
        - Drug match: 40% weight
        - Symptom similarity: 40% weight
        - Demographics (age/gender): 15% weight
        - Time recency: 5% weight
        
        Args:
            new_case (dict): New case with keys:
                - drug_name, symptoms, age, gender, created_at
            existing_case (Patient obj): Existing patient record
            
        Returns:
            dict: {
                'similarity_score': float (0-1),
                'breakdown': dict with individual scores,
                'is_match': bool (True if above threshold)
            }
        """
        scores = {}
        
        # 1. DRUG NAME MATCHING (40% weight) - Exact match is crucial
        drug_similarity = 1.0 if (
            new_case.get('drug_name', '').lower() == 
            existing_case.drug_name.lower()
        ) else 0.0
        scores['drug'] = drug_similarity
        
        # 2. SYMPTOM SIMILARITY (40% weight) - Fuzzy match
        new_symptoms = new_case.get('symptoms', '')
        existing_symptoms = existing_case.symptoms or ''
        symptom_similarity = self.calculate_text_similarity(
            new_symptoms, 
            existing_symptoms
        )
        scores['symptoms'] = symptom_similarity
        
        # 3. DEMOGRAPHICS MATCHING (15% weight)
        age_similarity = self.calculate_age_similarity(
            new_case.get('age'),
            existing_case.age
        )
        
        # Gender match - handle None values safely
        new_gender = (new_case.get('gender') or '').lower()
        existing_gender = (existing_case.gender or '').lower()
        gender_match = new_gender == existing_gender if (new_gender and existing_gender) else True
        gender_similarity = 1.0 if gender_match else 0.5
        
        demo_similarity = (age_similarity * 0.5) + (gender_similarity * 0.5)
        scores['demographics'] = demo_similarity
        
        # 4. TIME RECENCY (5% weight) - Recent cases are more likely duplicates
        time_similarity = 1.0
        if existing_case.created_at:
            days_diff = (datetime.utcnow() - existing_case.created_at).days
            # Max 30 days window; older cases less likely to be duplicates
            if days_diff <= 30:
                time_similarity = 1.0 - (days_diff / 30)
            else:
                time_similarity = 0.2
        scores['time'] = time_similarity
        
        # WEIGHTED SCORE CALCULATION
        weighted_score = (
            (drug_similarity * 0.40) +
            (symptom_similarity * 0.40) +
            (demo_similarity * 0.15) +
            (time_similarity * 0.05)
        )
        
        is_match = weighted_score >= self.threshold
        
        return {
            'similarity_score': round(weighted_score, 3),
            'breakdown': {
                'drug': round(drug_similarity, 3),
                'symptoms': round(symptom_similarity, 3),
                'demographics': round(demo_similarity, 3),
                'time_recency': round(time_similarity, 3)
            },
            'is_match': is_match,
            'confidence': self._get_confidence_level(weighted_score)
        }
    
    def _get_confidence_level(self, score):
        """Convert score to confidence level"""
        if score >= 0.9:
            return 'Very High'
        elif score >= 0.75:
            return 'High'
        elif score >= 0.6:
            return 'Medium'
        elif score >= 0.4:
            return 'Low'
        else:
            return 'Very Low'
    
    def find_matching_cases(self, new_case, existing_cases, limit=5):
        """
        Find all matching cases for a new case
        
        Args:
            new_case (dict): New case data
            existing_cases (list): List of existing Patient objects
            limit (int): Max number of matches to return
            
        Returns:
            dict: {
                'matches': list of dicts with case details and scores,
                'has_exact_match': bool,
                'recommendation': str (ACCEPT/REVIEW/DISCARD)
            }
        """
        if not existing_cases:
            return {
                'matches': [],
                'total_matches': 0,
                'has_exact_match': False,
                'recommendation': 'ACCEPT - No existing cases to match',
                'top_match': None
            }
        
        # Calculate similarity for all existing cases
        similarities = []
        for existing_case in existing_cases:
            result = self.calculate_case_similarity(new_case, existing_case)
            similarities.append({
                'case_id': existing_case.id,
                'patient_name': existing_case.name,
                'drug_name': existing_case.drug_name,
                'symptoms': existing_case.symptoms,
                'age': existing_case.age,
                'gender': existing_case.gender,
                'created_at': existing_case.created_at.isoformat(),
                **result
            })
        
        # Sort by similarity score descending
        similarities.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        # Filter matches above threshold
        matches = [s for s in similarities if s['is_match']]
        
        # Determine recommendation
        if len(matches) > 0:
            top_score = matches[0]['similarity_score']
            if top_score >= 0.90:
                recommendation = 'DISCARD - Very likely duplicate'
            elif top_score >= 0.80:
                recommendation = 'REVIEW - Likely duplicate, review before accepting'
            else:
                recommendation = 'REVIEW - Possible duplicate'
        else:
            recommendation = 'ACCEPT - No matching cases found'
        
        return {
            'matches': matches[:limit],
            'total_matches': len(matches),
            'has_exact_match': len(matches) > 0 and matches[0]['similarity_score'] >= 0.90,
            'recommendation': recommendation,
            'top_match': matches[0] if matches else None
        }


# Utility functions for use in app.py

def match_new_case(new_case_data, existing_patients, threshold=0.75):
    """
    Convenience function to match a new case against existing patients
    
    Args:
        new_case_data (dict): New case with drug_name, symptoms, age, gender, etc
        existing_patients (list): List of Patient objects
        threshold (float): Matching threshold (0-1)
        
    Returns:
        dict: Matching results with recommendations
    """
    engine = CaseMatchingEngine(threshold=threshold)
    return engine.find_matching_cases(new_case_data, existing_patients)


def should_accept_case(match_result, auto_discard_threshold=0.90):
    """
    Decide whether to auto-accept, review, or discard a case
    
    Args:
        match_result (dict): Result from find_matching_cases()
        auto_discard_threshold (float): Score above which to auto-discard
        
    Returns:
        dict: {
            'action': 'ACCEPT'|'REVIEW'|'DISCARD',
            'reason': str,
            'match_details': dict
        }
    """
    if not match_result['matches']:
        return {
            'action': 'ACCEPT',
            'reason': 'No matching cases found',
            'match_details': None
        }
    
    top_match = match_result['top_match']
    score = top_match['similarity_score']
    
    if score >= auto_discard_threshold:
        return {
            'action': 'DISCARD',
            'reason': f'Very high similarity ({score:.1%}) with case {top_match["case_id"]}',
            'match_details': top_match
        }
    elif score >= 0.80:
        return {
            'action': 'REVIEW',
            'reason': f'High similarity ({score:.1%}) with case {top_match["case_id"]} - manual review recommended',
            'match_details': top_match
        }
    else:
        return {
            'action': 'ACCEPT',
            'reason': f'Low similarity ({score:.1%}) - treating as new case',
            'match_details': top_match
        }
