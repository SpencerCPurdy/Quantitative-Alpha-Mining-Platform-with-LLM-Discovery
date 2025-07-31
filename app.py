"""
Quantitative Alpha Mining Platform with LLM Discovery
Author: Spencer Purdy
Description: A sophisticated platform that leverages LLMs to discover and evaluate alpha factors,
             combining classical quantitative approaches with modern ML techniques for comprehensive
             market analysis and portfolio construction.
"""

# Install required packages
# !pip install -q transformers torch numpy pandas scikit-learn plotly gradio yfinance ta scipy statsmodels openai seaborn

# Core imports
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime, timedelta
import gradio as gr
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import random
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict
import warnings
import os
import openai
warnings.filterwarnings('ignore')

# Statistical and ML imports
from scipy import stats
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm

# Technical analysis
import ta

# Transformers for NLP
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

# Constants for the trading system
RISK_FREE_RATE = 0.02
TRANSACTION_COST = 0.001  # 10 basis points
REBALANCE_FREQUENCY = 20  # Trading days
MIN_FACTOR_IC = 0.02  # Minimum Information Coefficient threshold
MAX_FACTOR_CORRELATION = 0.7  # Maximum correlation between factors

@dataclass
class AlphaFactor:
    """Data class representing an alpha factor"""
    name: str
    formula: str
    category: str  # 'price', 'volume', 'fundamental', 'alternative'
    lookback_period: int
    ic_score: float = 0.0
    sharpe_ratio: float = 0.0
    turnover: float = 0.0
    decay_rate: float = 0.0
    regime_performance: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MarketRegime:
    """Data class for market regime identification"""
    regime_type: str  # 'trending_up', 'trending_down', 'mean_reverting', 'volatile'
    confidence: float
    characteristics: Dict[str, float]
    start_date: datetime
    end_date: Optional[datetime] = None

class ClassicalAlphaFactors:
    """Implementation of classical alpha factors inspired by WorldQuant's 101 Alphas"""

    @staticmethod
    def safe_rank(series: pd.Series) -> pd.Series:
        """Safely rank a series handling NaN values"""
        return series.rank(pct=True, na_option='keep')

    @staticmethod
    def safe_rolling(series: pd.Series, window: int, func: str = 'mean') -> pd.Series:
        """Safely apply rolling window operations"""
        if len(series) < window:
            return pd.Series(np.nan, index=series.index)

        if func == 'mean':
            return series.rolling(window, min_periods=1).mean()
        elif func == 'std':
            return series.rolling(window, min_periods=1).std()
        elif func == 'max':
            return series.rolling(window, min_periods=1).max()
        elif func == 'min':
            return series.rolling(window, min_periods=1).min()
        elif func == 'sum':
            return series.rolling(window, min_periods=1).sum()
        else:
            return series.rolling(window, min_periods=1).mean()

    @staticmethod
    def alpha_001(data: pd.DataFrame) -> pd.Series:
        """Alpha#001: Momentum-based factor with volatility adjustment"""
        try:
            returns = data['close'].pct_change().fillna(0)
            condition = returns < 0
            stddev = ClassicalAlphaFactors.safe_rolling(returns, 20, 'std').fillna(0.01)

            signed_power = pd.Series(
                np.where(condition, stddev ** 2, data['close'] ** 2),
                index=data.index
            )

            ts_argmax = signed_power.rolling(5, min_periods=1).apply(
                lambda x: x.argmax() if len(x) > 0 else 0
            )

            result = ClassicalAlphaFactors.safe_rank(ts_argmax) - 0.5
            return result.fillna(0)
        except Exception as e:
            return pd.Series(0, index=data.index)

    @staticmethod
    def alpha_002(data: pd.DataFrame) -> pd.Series:
        """Alpha#002: Volume-price correlation factor"""
        try:
            # Ensure no division by zero
            data_safe = data.copy()
            data_safe['volume'] = data_safe['volume'].replace(0, 1)
            data_safe['open'] = data_safe['open'].replace(0, data_safe['close'])

            log_volume_delta = np.log(data_safe['volume']).diff(2).fillna(0)
            price_change_ratio = ((data_safe['close'] - data_safe['open']) / data_safe['open']).fillna(0)

            rank1 = ClassicalAlphaFactors.safe_rank(log_volume_delta)
            rank2 = ClassicalAlphaFactors.safe_rank(price_change_ratio)

            correlation = rank1.rolling(6, min_periods=1).corr(rank2)
            return (-1 * correlation).fillna(0)
        except Exception as e:
            return pd.Series(0, index=data.index)

    @staticmethod
    def alpha_003(data: pd.DataFrame) -> pd.Series:
        """Alpha#003: Open-volume correlation"""
        try:
            rank_open = ClassicalAlphaFactors.safe_rank(data['open'])
            rank_volume = ClassicalAlphaFactors.safe_rank(data['volume'])

            correlation = rank_open.rolling(10, min_periods=1).corr(rank_volume)
            return (-1 * correlation).fillna(0)
        except Exception as e:
            return pd.Series(0, index=data.index)

    @staticmethod
    def alpha_004(data: pd.DataFrame) -> pd.Series:
        """Alpha#004: Low price time series rank"""
        try:
            rank_low = ClassicalAlphaFactors.safe_rank(data['low'])
            ts_rank = rank_low.rolling(9, min_periods=1).apply(
                lambda x: ClassicalAlphaFactors.safe_rank(pd.Series(x)).iloc[-1] if len(x) > 0 else 0.5
            )
            return (-1 * ts_rank).fillna(0)
        except Exception as e:
            return pd.Series(0, index=data.index)

    @staticmethod
    def alpha_005(data: pd.DataFrame) -> pd.Series:
        """Alpha#005: VWAP-based factor"""
        try:
            # Calculate VWAP safely
            data_safe = data.copy()
            data_safe['volume'] = data_safe['volume'].replace(0, 1)

            vwap = (data_safe['close'] * data_safe['volume']).cumsum() / data_safe['volume'].cumsum()
            vwap_ma = ClassicalAlphaFactors.safe_rolling(vwap, 10, 'mean')

            rank1 = ClassicalAlphaFactors.safe_rank(data_safe['open'] - vwap_ma)
            rank2 = np.abs(ClassicalAlphaFactors.safe_rank(data_safe['close'] - vwap))

            result = rank1 * (-1 * rank2)
            return result.fillna(0)
        except Exception as e:
            return pd.Series(0, index=data.index)

    @staticmethod
    def alpha_006(data: pd.DataFrame) -> pd.Series:
        """Alpha#006: Open-volume correlation"""
        try:
            correlation = data['open'].rolling(10, min_periods=1).corr(data['volume'])
            return (-1 * correlation).fillna(0)
        except Exception as e:
            return pd.Series(0, index=data.index)

    @staticmethod
    def alpha_007(data: pd.DataFrame) -> pd.Series:
        """Alpha#007: Volume-based momentum"""
        try:
            adv20 = ClassicalAlphaFactors.safe_rolling(data['volume'], 20, 'mean')
            condition = adv20 < data['volume']

            close_delta = data['close'].diff(7).fillna(0)
            abs_delta = np.abs(close_delta)

            ts_rank = abs_delta.rolling(60, min_periods=1).apply(
                lambda x: ClassicalAlphaFactors.safe_rank(pd.Series(x)).iloc[-1] if len(x) > 0 else 0.5
            )

            result = pd.Series(
                np.where(condition, -1 * ts_rank * np.sign(close_delta), -1),
                index=data.index
            )
            return result.fillna(0)
        except Exception as e:
            return pd.Series(0, index=data.index)

    @staticmethod
    def alpha_008(data: pd.DataFrame) -> pd.Series:
        """Alpha#008: Open-return product factor"""
        try:
            returns = data['close'].pct_change().fillna(0)
            sum_open = ClassicalAlphaFactors.safe_rolling(data['open'], 5, 'sum')
            sum_returns = ClassicalAlphaFactors.safe_rolling(returns, 5, 'sum')

            product = sum_open * sum_returns
            delayed_product = product.shift(10).fillna(method='bfill')

            result = -1 * ClassicalAlphaFactors.safe_rank(product - delayed_product)
            return result.fillna(0)
        except Exception as e:
            return pd.Series(0, index=data.index)

    @staticmethod
    def alpha_009(data: pd.DataFrame) -> pd.Series:
        """Alpha#009: Close delta conditional factor"""
        try:
            close_delta = data['close'].diff(1).fillna(0)
            ts_min = ClassicalAlphaFactors.safe_rolling(close_delta, 5, 'min')
            ts_max = ClassicalAlphaFactors.safe_rolling(close_delta, 5, 'max')

            condition1 = ts_min > 0
            condition2 = ts_max < 0

            result = pd.Series(
                np.where(condition1, close_delta,
                        np.where(condition2, close_delta, -1 * close_delta)),
                index=data.index
            )
            return result.fillna(0)
        except Exception as e:
            return pd.Series(0, index=data.index)

    @staticmethod
    def alpha_010(data: pd.DataFrame) -> pd.Series:
        """Alpha#010: Ranked version of alpha_009"""
        try:
            close_delta = data['close'].diff(1).fillna(0)
            ts_min = ClassicalAlphaFactors.safe_rolling(close_delta, 4, 'min')
            ts_max = ClassicalAlphaFactors.safe_rolling(close_delta, 4, 'max')

            condition1 = ts_min > 0
            condition2 = ts_max < 0

            raw_result = pd.Series(
                np.where(condition1, close_delta,
                        np.where(condition2, close_delta, -1 * close_delta)),
                index=data.index
            )

            result = ClassicalAlphaFactors.safe_rank(raw_result)
            return result.fillna(0)
        except Exception as e:
            return pd.Series(0, index=data.index)

    @staticmethod
    def get_all_classical_factors() -> List[callable]:
        """Return list of all classical alpha factor functions"""
        return [
            ClassicalAlphaFactors.alpha_001,
            ClassicalAlphaFactors.alpha_002,
            ClassicalAlphaFactors.alpha_003,
            ClassicalAlphaFactors.alpha_004,
            ClassicalAlphaFactors.alpha_005,
            ClassicalAlphaFactors.alpha_006,
            ClassicalAlphaFactors.alpha_007,
            ClassicalAlphaFactors.alpha_008,
            ClassicalAlphaFactors.alpha_009,
            ClassicalAlphaFactors.alpha_010
        ]

class LLMAlphaGenerator:
    """Generate novel alpha factors using OpenAI's GPT models"""

    def __init__(self, api_key: str = None):
        self.api_key = api_key
        if self.api_key:
            openai.api_key = self.api_key

        self.operators = ['rank', 'ts_rank', 'ts_sum', 'ts_mean', 'ts_std', 'ts_max', 'ts_min',
                         'correlation', 'covariance', 'delta', 'delay', 'log', 'sign', 'abs']
        self.variables = ['open', 'high', 'low', 'close', 'volume', 'returns', 'vwap']
        self.generated_factors = []

    def generate_llm_factor(self, market_context: Dict[str, Any], category: str) -> Tuple[str, str]:
        """Generate a novel alpha factor formula using OpenAI's GPT model"""

        # If no API key, use fallback method
        if not self.api_key:
            return self._generate_fallback_factor(category)

        # Create prompt for the LLM
        prompt = f"""You are a quantitative researcher creating novel alpha factors for trading.

Market Context:
- Current Regime: {market_context.get('regime', 'unknown')}
- Average Volatility: {market_context.get('volatility', 0.02):.1%}
- Trend Strength: {market_context.get('trend_strength', 0.5):.1%}

Task: Generate a novel alpha factor formula for the '{category}' category.

Available operators: {', '.join(self.operators)}
Available variables: {', '.join(self.variables)}

Requirements:
1. The formula must be executable Python code using pandas operations
2. Use time-series operators (ts_*) with appropriate lookback periods
3. The factor should capture {category} characteristics
4. Include rank transformations to make the factor cross-sectionally comparable
5. The formula should be between 50-150 characters

Examples of good alpha factors:
- rank(ts_sum(returns, 20)) * rank(volume / ts_mean(volume, 20))
- -1 * correlation(rank(close), rank(volume), 10)
- sign(returns) * ts_std(returns, 20) / ts_mean(abs(returns), 20)

Generate ONE formula that captures {category} patterns. Return ONLY the formula, no explanation."""

        try:
            # Call OpenAI API
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a quantitative finance expert specializing in alpha factor research."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=150
            )

            formula = response.choices[0].message.content.strip()

            # Validate the formula
            if self.validate_formula(formula):
                name = f"LLM_{category}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                self.generated_factors.append({'name': name, 'formula': formula, 'category': category})
                return name, formula
            else:
                return self._generate_fallback_factor(category)

        except Exception as e:
            print(f"LLM generation error: {e}")
            return self._generate_fallback_factor(category)

    def _generate_fallback_factor(self, category: str) -> Tuple[str, str]:
        """Generate a fallback factor if LLM generation fails"""
        templates = {
            'momentum': "rank(ts_sum(returns, 20)) * rank(volume / ts_mean(volume, 20))",
            'mean_reversion': "-1 * (close - ts_mean(close, 20)) / ts_std(close, 20)",
            'volatility': "ts_std(returns, 20) / ts_mean(abs(returns), 20)",
            'microstructure': "(high - low) / (high + low) * rank(volume)",
            'price': "rank(close / ts_max(high, 20))",
            'volume': "rank(volume / ts_mean(volume, 50))",
            'fundamental': "rank(close * volume / ts_sum(volume, 10))",
            'alternative': "rank(ts_std(volume, 10) / ts_mean(volume, 30))"
        }

        formula = templates.get(category, templates['momentum'])
        name = f"Fallback_{category}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        return name, formula

    def validate_formula(self, formula: str) -> bool:
        """Validate that a formula is syntactically correct and safe"""
        try:
            # Check for balanced parentheses
            if formula.count('(') != formula.count(')'):
                return False

            # Check for dangerous operations
            dangerous_ops = ['eval', 'exec', 'import', '__', 'lambda', 'os', 'sys']
            for op in dangerous_ops:
                if op in formula:
                    return False

            # Check that it contains at least one operator and one variable
            has_operator = any(op in formula for op in self.operators)
            has_variable = any(var in formula for var in self.variables)

            return has_operator and has_variable

        except:
            return False

    def evaluate_formula(self, formula: str, data: pd.DataFrame) -> pd.Series:
        """Safely evaluate a formula on market data"""
        try:
            # Prepare safe data
            safe_data = data.copy()
            safe_data['volume'] = safe_data['volume'].replace(0, 1)  # Avoid division by zero

            # Calculate derived variables
            returns = safe_data['close'].pct_change().fillna(0)
            vwap = (safe_data['close'] * safe_data['volume']).cumsum() / safe_data['volume'].cumsum()
            vwap = vwap.fillna(safe_data['close'])
            adv20 = safe_data['volume'].rolling(20, min_periods=1).mean()

            # Create evaluation context
            context = {
                'open': safe_data['open'],
                'high': safe_data['high'],
                'low': safe_data['low'],
                'close': safe_data['close'],
                'volume': safe_data['volume'],
                'returns': returns,
                'vwap': vwap,
                'adv20': adv20
            }

            # Define safe functions with error handling
            def safe_rank(x):
                return x.rank(pct=True, na_option='keep').fillna(0.5)

            def safe_ts_rank(x, n):
                return x.rolling(n, min_periods=1).apply(
                    lambda y: y.rank(pct=True).iloc[-1] if len(y) > 0 else 0.5
                ).fillna(0.5)

            def safe_ts_sum(x, n):
                return x.rolling(n, min_periods=1).sum().fillna(0)

            def safe_ts_mean(x, n):
                return x.rolling(n, min_periods=1).mean().fillna(x.fillna(0))

            def safe_ts_std(x, n):
                result = x.rolling(n, min_periods=1).std()
                return result.fillna(0.001)  # Small non-zero value

            def safe_ts_max(x, n):
                return x.rolling(n, min_periods=1).max().fillna(x.fillna(0))

            def safe_ts_min(x, n):
                return x.rolling(n, min_periods=1).min().fillna(x.fillna(0))

            def safe_correlation(x, y, n):
                return x.rolling(n, min_periods=1).corr(y).fillna(0)

            def safe_covariance(x, y, n):
                return x.rolling(n, min_periods=1).cov(y).fillna(0)

            def safe_delta(x, n):
                return x.diff(n).fillna(0)

            def safe_delay(x, n):
                return x.shift(n).fillna(method='bfill').fillna(0)

            def safe_log(x):
                return np.log(x.clip(lower=0.001))

            def safe_sign(x):
                return np.sign(x).fillna(0)

            def safe_abs(x):
                return np.abs(x).fillna(0)

            # Safe functions namespace
            safe_functions = {
                'rank': safe_rank,
                'ts_rank': safe_ts_rank,
                'ts_sum': safe_ts_sum,
                'ts_mean': safe_ts_mean,
                'ts_std': safe_ts_std,
                'ts_max': safe_ts_max,
                'ts_min': safe_ts_min,
                'correlation': safe_correlation,
                'covariance': safe_covariance,
                'delta': safe_delta,
                'delay': safe_delay,
                'log': safe_log,
                'sign': safe_sign,
                'abs': safe_abs,
                'np': np,
                'pd': pd
            }

            # Combine context and functions
            eval_namespace = {**context, **safe_functions}

            # Evaluate formula with restricted namespace
            result = eval(formula, {"__builtins__": {}}, eval_namespace)

            # Convert to Series if needed
            if not isinstance(result, pd.Series):
                result = pd.Series(result, index=data.index)

            # Final safety checks
            result = result.replace([np.inf, -np.inf], 0)
            result = result.fillna(0)

            return result

        except Exception as e:
            print(f"Error evaluating formula '{formula}': {e}")
            # Return a neutral factor (zeros) on error
            return pd.Series(0, index=data.index)

class AlternativeDataPipeline:
    """Extract sentiment scores from alternative data sources"""

    def __init__(self):
        # Initialize sentiment analysis model
        try:
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                device=-1  # CPU
            )
        except:
            # Fallback to a simpler model if FinBERT fails
            self.sentiment_analyzer = None

        # Simulated data sources
        self.data_sources = {
            'earnings_calls': self._generate_earnings_call_snippets,
            'sec_filings': self._generate_sec_filing_snippets,
            'news': self._generate_news_snippets,
            'social_media': self._generate_social_media_snippets
        }

    def _generate_earnings_call_snippets(self) -> List[str]:
        """Generate simulated earnings call transcripts"""
        positive_phrases = [
            "We exceeded our revenue guidance for the quarter with strong performance across all segments",
            "Our strategic initiatives are yielding positive results with improved margins",
            "Customer acquisition costs have decreased while lifetime value continues to grow",
            "We're seeing strong demand for our products in emerging markets",
            "Our R&D investments are beginning to show promising returns"
        ]

        negative_phrases = [
            "We faced headwinds in our core markets due to increased competition",
            "Supply chain disruptions continue to impact our margins",
            "We're revising our guidance downward for the upcoming quarter",
            "Customer churn rates have increased beyond our expectations",
            "Regulatory challenges in key markets are affecting our expansion plans"
        ]

        neutral_phrases = [
            "We maintained our market position despite challenging conditions",
            "Our performance was in line with analyst expectations",
            "We continue to execute on our long-term strategic plan",
            "Market conditions remain mixed with both opportunities and challenges",
            "We're monitoring the situation closely and will adjust as needed"
        ]

        # Mix phrases based on market conditions
        market_sentiment = random.choice(['positive', 'negative', 'neutral'])

        if market_sentiment == 'positive':
            return random.sample(positive_phrases, min(3, len(positive_phrases))) + \
                   random.sample(neutral_phrases, min(1, len(neutral_phrases)))
        elif market_sentiment == 'negative':
            return random.sample(negative_phrases, min(3, len(negative_phrases))) + \
                   random.sample(neutral_phrases, min(1, len(neutral_phrases)))
        else:
            return random.sample(neutral_phrases, min(2, len(neutral_phrases))) + \
                   random.sample(positive_phrases, min(1, len(positive_phrases))) + \
                   random.sample(negative_phrases, min(1, len(negative_phrases)))

    def _generate_sec_filing_snippets(self) -> List[str]:
        """Generate simulated SEC filing excerpts"""
        risk_factors = [
            "The company faces increased cybersecurity risks that could materially affect operations",
            "Changes in interest rates may adversely impact our financial condition",
            "We depend on key personnel whose loss could harm our business",
            "Intense competition in our industry may result in reduced market share",
            "Economic uncertainty could reduce demand for our products and services"
        ]

        positive_disclosures = [
            "We have secured long-term contracts with several major customers",
            "Our patent portfolio provides strong competitive advantages",
            "Recent acquisitions are expected to be accretive to earnings",
            "We maintain a strong balance sheet with minimal debt",
            "Our diversified revenue streams provide resilience against market volatility"
        ]

        return random.sample(risk_factors, min(2, len(risk_factors))) + \
               random.sample(positive_disclosures, min(2, len(positive_disclosures)))

    def _generate_news_snippets(self) -> List[str]:
        """Generate simulated financial news headlines"""
        headlines = [
            "Company announces breakthrough technology in core product line",
            "Analysts upgrade stock following strong quarterly results",
            "New CEO brings fresh perspective and growth strategy",
            "Competitor's product recall may benefit company's market share",
            "Industry report shows growing demand for company's services",
            "Regulatory approval received for expansion into new markets",
            "Company faces lawsuit over alleged patent infringement",
            "Major customer switches to competitor's platform",
            "Economic indicators suggest challenging environment ahead"
        ]

        return random.sample(headlines, min(5, len(headlines)))

    def _generate_social_media_snippets(self) -> List[str]:
        """Generate simulated social media sentiment"""
        posts = [
            "Love the new features in the latest product update! #innovation",
            "Customer service has really improved lately, impressed!",
            "Stock looking oversold here, might be a buying opportunity",
            "Disappointed with the recent earnings miss, concerning trend",
            "Management seems to be making all the right moves",
            "Product quality has declined, considering alternatives",
            "Excited about the company's expansion plans",
            "Valuation seems stretched at current levels"
        ]

        return random.sample(posts, min(4, len(posts)))

    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of a single text"""
        if self.sentiment_analyzer is None:
            # Fallback sentiment analysis
            positive_words = ['strong', 'exceed', 'growth', 'positive', 'improve', 'breakthrough']
            negative_words = ['decline', 'loss', 'risk', 'challenge', 'lawsuit', 'disappoint']

            text_lower = text.lower()
            pos_count = sum(1 for word in positive_words if word in text_lower)
            neg_count = sum(1 for word in negative_words if word in text_lower)

            if pos_count > neg_count:
                return {'label': 'positive', 'score': 0.7}
            elif neg_count > pos_count:
                return {'label': 'negative', 'score': 0.7}
            else:
                return {'label': 'neutral', 'score': 0.5}

        try:
            result = self.sentiment_analyzer(text[:512])[0]
            return result
        except:
            return {'label': 'neutral', 'score': 0.5}

    def extract_sentiment_scores(self, source: str = 'all') -> Dict[str, Dict[str, float]]:
        """Extract sentiment scores from specified data source"""
        sentiment_scores = {}

        if source == 'all':
            sources_to_analyze = self.data_sources.keys()
        else:
            sources_to_analyze = [source] if source in self.data_sources else []

        for src in sources_to_analyze:
            snippets = self.data_sources[src]()

            # Analyze each snippet
            positive_count = 0
            negative_count = 0
            total_score = 0

            for snippet in snippets:
                try:
                    result = self.analyze_sentiment(snippet)

                    if result['label'] == 'positive':
                        positive_count += 1
                        total_score += result['score']
                    elif result['label'] == 'negative':
                        negative_count += 1
                        total_score -= result['score']

                except:
                    continue

            # Calculate aggregate sentiment
            if len(snippets) > 0:
                sentiment_scores[src] = {
                    'positive_ratio': positive_count / len(snippets),
                    'negative_ratio': negative_count / len(snippets),
                    'net_sentiment': total_score / len(snippets),
                    'snippets_analyzed': len(snippets)
                }
            else:
                sentiment_scores[src] = {
                    'positive_ratio': 0,
                    'negative_ratio': 0,
                    'net_sentiment': 0,
                    'snippets_analyzed': 0
                }

        return sentiment_scores

    def create_sentiment_alpha_factors(self, sentiment_scores: Dict[str, Dict[str, float]]) -> List[AlphaFactor]:
        """Create alpha factors based on sentiment scores"""
        factors = []

        # Earnings call sentiment factor
        if 'earnings_calls' in sentiment_scores:
            factor = AlphaFactor(
                name="sentiment_earnings_momentum",
                formula="earnings_sentiment * volume_ratio",
                category="alternative",
                lookback_period=20,
                metadata={'sentiment_data': sentiment_scores['earnings_calls']}
            )
            factors.append(factor)

        # News sentiment factor
        if 'news' in sentiment_scores:
            factor = AlphaFactor(
                name="sentiment_news_reversal",
                formula="-1 * news_sentiment * (close - ma20) / std20",
                category="alternative",
                lookback_period=20,
                metadata={'sentiment_data': sentiment_scores['news']}
            )
            factors.append(factor)

        # Composite sentiment factor
        if len(sentiment_scores) > 1:
            avg_sentiment = np.mean([s['net_sentiment'] for s in sentiment_scores.values()])

            factor = AlphaFactor(
                name="sentiment_composite",
                formula="composite_sentiment * rank(volume)",
                category="alternative",
                lookback_period=10,
                metadata={
                    'avg_sentiment': avg_sentiment,
                    'sources': list(sentiment_scores.keys())
                }
            )
            factors.append(factor)

        return factors

class MarketRegimeDetector:
    """Detect market regimes using statistical methods"""

    def __init__(self):
        self.regime_history = []
        self.current_regime = None

    def detect_regime(self, data: pd.DataFrame, lookback: int = 60) -> MarketRegime:
        """Detect current market regime"""

        # Ensure we have enough data
        if len(data) < 20:  # Minimum required
            return MarketRegime(
                regime_type='volatile',
                confidence=0.5,
                characteristics={
                    'trend_strength': 0,
                    'volatility': 0.02,
                    'hurst_exponent': 0.5,
                    'volume_trend': 0,
                    'avg_return': 0
                },
                start_date=data.index[0] if len(data) > 0 else datetime.now()
            )

        if len(data) < lookback:
            lookback = len(data)

        # Calculate features
        returns = data['close'].pct_change().fillna(0)
        recent_returns = returns.iloc[-lookback:]

        # Trend strength
        trend_strength = self._calculate_trend_strength(data['close'].iloc[-lookback:])

        # Volatility
        volatility = recent_returns.std() * np.sqrt(252)

        # Mean reversion test
        hurst_exponent = self._calculate_hurst_exponent(data['close'].iloc[-lookback:])

        # Volume patterns
        volume_data = data['volume'].iloc[-lookback:].fillna(0)
        if len(volume_data) > 1:
            try:
                volume_trend = np.polyfit(range(len(volume_data)), volume_data, 1)[0]
            except:
                volume_trend = 0
        else:
            volume_trend = 0

        # Classify regime
        avg_return = recent_returns.mean()

        if trend_strength > 0.6 and avg_return > 0.001:
            regime_type = 'trending_up'
        elif trend_strength > 0.6 and avg_return < -0.001:
            regime_type = 'trending_down'
        elif hurst_exponent < 0.45:
            regime_type = 'mean_reverting'
        else:
            regime_type = 'volatile'

        # Calculate confidence
        confidence = self._calculate_regime_confidence(
            trend_strength, volatility, hurst_exponent
        )

        regime = MarketRegime(
            regime_type=regime_type,
            confidence=confidence,
            characteristics={
                'trend_strength': trend_strength,
                'volatility': volatility,
                'hurst_exponent': hurst_exponent,
                'volume_trend': volume_trend,
                'avg_return': avg_return
            },
            start_date=data.index[-lookback] if lookback <= len(data) else data.index[0]
        )

        self.current_regime = regime
        return regime

    def _calculate_trend_strength(self, prices: pd.Series) -> float:
        """Calculate trend strength using R-squared of linear regression"""
        try:
            if len(prices) < 2:
                return 0

            x = np.arange(len(prices))
            y = prices.values

            # Remove NaN values
            mask = ~np.isnan(y)
            if mask.sum() < 2:
                return 0

            x = x[mask]
            y = y[mask]

            # Normalize
            x_std = x.std()
            y_std = y.std()

            if x_std == 0 or y_std == 0:
                return 0

            x = (x - x.mean()) / x_std
            y = (y - y.mean()) / y_std

            # Linear regression
            slope, intercept = np.polyfit(x, y, 1)
            y_pred = slope * x + intercept

            # R-squared
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)

            if ss_tot == 0:
                return 0

            r_squared = 1 - (ss_res / ss_tot)
            return abs(r_squared)

        except:
            return 0

    def _calculate_hurst_exponent(self, prices: pd.Series) -> float:
        """Calculate Hurst exponent for mean reversion detection"""
        try:
            if len(prices) < 20:
                return 0.5

            # Use a fixed set of lags
            max_lag = min(20, len(prices) // 2)
            lags = range(2, max_lag)

            # Calculate R/S for different lags
            rs_values = []

            for lag in lags:
                # Calculate returns
                returns = prices.pct_change(lag).dropna()

                if len(returns) < 2:
                    continue

                # Mean-adjusted series
                mean_returns = returns.mean()
                adjusted = returns - mean_returns

                # Cumulative sum
                cumsum = adjusted.cumsum()

                # Range
                R = cumsum.max() - cumsum.min()

                # Standard deviation
                S = returns.std()

                if S > 0 and R > 0:
                    rs_values.append(R / S)

            if len(rs_values) >= 2:
                # Log-log regression
                valid_lags = list(lags[:len(rs_values)])
                log_lags = np.log(valid_lags)
                log_rs = np.log(rs_values)

                # Remove any inf or nan values
                mask = np.isfinite(log_lags) & np.isfinite(log_rs)
                if mask.sum() >= 2:
                    hurst, _ = np.polyfit(log_lags[mask], log_rs[mask], 1)
                    return max(0, min(1, hurst))  # Bound between 0 and 1

            return 0.5  # Random walk

        except:
            return 0.5

    def _calculate_regime_confidence(self, trend_strength: float,
                                   volatility: float, hurst: float) -> float:
        """Calculate confidence in regime classification"""
        # Base confidence
        confidence = 0.5

        # Strong trend
        if trend_strength > 0.7:
            confidence += 0.2
        elif trend_strength > 0.5:
            confidence += 0.1

        # Clear mean reversion or trending
        if abs(hurst - 0.5) > 0.2:
            confidence += 0.15
        elif abs(hurst - 0.5) > 0.1:
            confidence += 0.075

        # Volatility consistency
        if 0.1 < volatility < 0.4:  # Normal range
            confidence += 0.15
        elif 0.05 < volatility < 0.5:
            confidence += 0.075

        return min(confidence, 1.0)

class FactorEvaluator:
    """Evaluate alpha factors using various metrics"""

    def __init__(self):
        self.evaluation_history = defaultdict(list)

    def calculate_information_coefficient(self, factor_values: pd.Series,
                                        forward_returns: pd.Series) -> float:
        """Calculate Information Coefficient (IC)"""
        try:
            # Remove NaN values
            mask = factor_values.notna() & forward_returns.notna()
            clean_factor = factor_values[mask]
            clean_returns = forward_returns[mask]

            if len(clean_factor) < 20:  # Need minimum observations
                return 0.0

            # Check for zero variance
            if clean_factor.std() == 0 or clean_returns.std() == 0:
                return 0.0

            # Rank correlation (Spearman)
            ic = stats.spearmanr(clean_factor, clean_returns)[0]

            return ic if not np.isnan(ic) else 0.0

        except:
            return 0.0

    def calculate_factor_turnover(self, factor_values: pd.Series,
                                 rebalance_freq: int = 20) -> float:
        """Calculate factor turnover"""
        try:
            if len(factor_values) < rebalance_freq * 2:
                return 0.0

            # Get factor ranks
            ranks = factor_values.rank(pct=True, na_option='keep').fillna(0.5)

            # Calculate portfolio positions (top/bottom quintiles)
            long_positions = ranks > 0.8
            short_positions = ranks < 0.2

            # Calculate turnover at rebalance points
            turnover_rates = []

            for i in range(rebalance_freq, len(ranks), rebalance_freq):
                prev_long = long_positions.iloc[i-rebalance_freq]
                curr_long = long_positions.iloc[i]

                prev_short = short_positions.iloc[i-rebalance_freq]
                curr_short = short_positions.iloc[i]

                # Turnover is the fraction of positions that changed
                long_turnover = (prev_long != curr_long).mean()
                short_turnover = (prev_short != curr_short).mean()

                turnover_rates.append((long_turnover + short_turnover) / 2)

            return np.mean(turnover_rates) if turnover_rates else 0.0

        except:
            return 0.0

    def calculate_factor_decay(self, factor: AlphaFactor,
                              market_data: pd.DataFrame,
                              max_lag: int = 20) -> Dict[int, float]:
        """Calculate IC decay over different prediction horizons"""
        ic_by_lag = {}

        try:
            # Evaluate factor to get values
            factor_values = self._get_factor_values(factor, market_data)

            # Calculate IC for different forward return periods
            for lag in range(1, min(max_lag + 1, len(market_data) - 1)):
                forward_returns = market_data['close'].pct_change(lag).shift(-lag)
                ic = self.calculate_information_coefficient(factor_values, forward_returns)
                ic_by_lag[lag] = ic

        except:
            # Return default decay
            for lag in range(1, max_lag + 1):
                ic_by_lag[lag] = 0.0

        return ic_by_lag

    def _get_factor_values(self, factor: AlphaFactor, market_data: pd.DataFrame) -> pd.Series:
        """Get factor values from formula or function"""
        try:
            if isinstance(factor.formula, str):
                if 'sentiment' in factor.name:
                    # For sentiment factors, create values based on metadata
                    if 'sentiment_data' in factor.metadata:
                        sentiment = factor.metadata['sentiment_data'].get('net_sentiment', 0)
                        # Create factor values that incorporate sentiment
                        base_values = market_data['volume'] / market_data['volume'].rolling(20, min_periods=1).mean()
                        factor_values = base_values * (1 + sentiment)
                    else:
                        # Generate random sentiment-like factor
                        factor_values = pd.Series(
                            np.random.normal(0, 0.1, len(market_data)),
                            index=market_data.index
                        ).cumsum() * 0.01
                else:
                    # Evaluate formula
                    llm_gen = LLMAlphaGenerator()
                    factor_values = llm_gen.evaluate_formula(factor.formula, market_data)
            else:
                # Classical factor (callable)
                factor_values = factor.formula(market_data)

            # Clean up values
            factor_values = factor_values.replace([np.inf, -np.inf], np.nan)
            factor_values = factor_values.fillna(0)

            return factor_values

        except:
            # Return neutral factor on error
            return pd.Series(0, index=market_data.index)

    def evaluate_factor_performance(self, factor: AlphaFactor,
                                  market_data: pd.DataFrame,
                                  regime: Optional[MarketRegime] = None) -> Dict[str, float]:
        """Comprehensive factor performance evaluation"""
        try:
            # Get factor values
            factor_values = self._get_factor_values(factor, market_data)

            # Forward returns
            forward_returns = market_data['close'].pct_change().shift(-1)

            # Calculate metrics
            ic = self.calculate_information_coefficient(factor_values, forward_returns)
            turnover = self.calculate_factor_turnover(factor_values)

            # Sharpe ratio of factor portfolio
            factor_portfolio_returns = self._calculate_factor_portfolio_returns(
                factor_values, forward_returns
            )
            sharpe = self._calculate_sharpe_ratio(factor_portfolio_returns)

            # Max drawdown
            max_dd = self._calculate_max_drawdown(factor_portfolio_returns)

            # Hit rate
            hit_rate = (factor_portfolio_returns > 0).mean() if len(factor_portfolio_returns) > 0 else 0.5

            metrics = {
                'ic': ic,
                'turnover': turnover,
                'sharpe_ratio': sharpe,
                'max_drawdown': max_dd,
                'hit_rate': hit_rate
            }

            # Store in history
            self.evaluation_history[factor.name].append({
                'timestamp': datetime.now(),
                'metrics': metrics,
                'regime': regime.regime_type if regime else 'unknown'
            })

            return metrics

        except:
            # Return default metrics on error
            return {
                'ic': 0.0,
                'turnover': 0.5,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.1,
                'hit_rate': 0.5
            }

    def _calculate_factor_portfolio_returns(self, factor_values: pd.Series,
                                          forward_returns: pd.Series) -> pd.Series:
        """Calculate returns of long-short portfolio based on factor"""
        try:
            # Rank stocks by factor
            ranks = factor_values.rank(pct=True, na_option='keep').fillna(0.5)

            # Long top quintile, short bottom quintile
            long_weight = (ranks > 0.8).astype(float)
            short_weight = (ranks < 0.2).astype(float)

            # Normalize weights
            long_sum = long_weight.sum()
            short_sum = short_weight.sum()

            if long_sum > 0:
                long_weight = long_weight / long_sum
            if short_sum > 0:
                short_weight = short_weight / short_sum

            # Portfolio returns
            portfolio_returns = (long_weight - short_weight) * forward_returns
            portfolio_returns = portfolio_returns.fillna(0)

            return portfolio_returns

        except:
            return pd.Series(0, index=forward_returns.index)

    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio"""
        try:
            if len(returns) < 2:
                return 0.0

            clean_returns = returns.dropna()
            if len(clean_returns) < 2:
                return 0.0

            excess_returns = clean_returns - RISK_FREE_RATE / 252

            if clean_returns.std() > 0:
                return np.sqrt(252) * excess_returns.mean() / clean_returns.std()
            else:
                return 0.0

        except:
            return 0.0

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        try:
            if len(returns) < 2:
                return 0.0

            # Calculate cumulative returns
            cum_returns = (1 + returns.fillna(0)).cumprod()

            # Calculate running maximum
            running_max = cum_returns.expanding().max()

            # Calculate drawdown
            drawdown = (cum_returns - running_max) / running_max

            # Return maximum drawdown (positive value)
            return abs(drawdown.min()) if len(drawdown) > 0 else 0.0

        except:
            return 0.0

class HierarchicalRiskParity:
    """Hierarchical Risk Parity portfolio construction"""

    def __init__(self):
        self.linkage_method = 'single'
        self.distance_metric = 'euclidean'

    def calculate_weights(self, returns: pd.DataFrame,
                         factor_scores: pd.DataFrame) -> pd.Series:
        """Calculate HRP weights for factors"""

        # Handle case with single factor or no data
        if returns.empty or len(returns.columns) == 0:
            return pd.Series()

        if len(returns.columns) == 1:
            return pd.Series(1.0, index=returns.columns)

        try:
            # Calculate correlation matrix
            corr_matrix = returns.corr()

            # Replace NaN values with 0
            corr_matrix = corr_matrix.fillna(0)

            # Ensure diagonal is 1
            np.fill_diagonal(corr_matrix.values, 1)

            # Calculate distance matrix
            dist_matrix = np.sqrt(2 * (1 - corr_matrix))

            # Perform hierarchical clustering
            condensed_dist = dist_matrix[np.triu_indices(len(dist_matrix), k=1)]
            linkage_matrix = self._tree_clustering(condensed_dist)

            # Get quasi-diagonal matrix
            quasi_diag = self._get_quasi_diag(linkage_matrix)

            # Calculate weights
            weights = self._get_recursive_bisection(
                returns.cov().fillna(0),
                quasi_diag
            )

            return pd.Series(weights, index=returns.columns)

        except:
            # Equal weights as fallback
            return pd.Series(1.0 / len(returns.columns), index=returns.columns)

    def _tree_clustering(self, dist_matrix: np.ndarray) -> np.ndarray:
        """Perform hierarchical clustering"""
        try:
            from scipy.cluster.hierarchy import linkage
            return linkage(dist_matrix, method=self.linkage_method)
        except:
            # Return dummy linkage matrix
            n = int((1 + np.sqrt(1 + 8 * len(dist_matrix))) / 2)
            return np.zeros((n-1, 4))

    def _get_quasi_diag(self, linkage_matrix: np.ndarray) -> List[int]:
        """Get quasi-diagonal matrix ordering"""
        try:
            from scipy.cluster.hierarchy import dendrogram

            # Get dendrogram
            dendro = dendrogram(linkage_matrix, no_plot=True)

            # Return ordering
            return dendro['leaves']
        except:
            # Return default ordering
            n = linkage_matrix.shape[0] + 1
            return list(range(n))

    def _get_recursive_bisection(self, cov: pd.DataFrame,
                                sort_idx: List[int]) -> np.ndarray:
        """Recursive bisection for weight calculation"""
        try:
            # Initialize weights
            weights = pd.Series(1, index=cov.index)

            # Recursive bisection
            items = [sort_idx]

            while len(items) > 0:
                # Pop item
                item = items.pop()

                if len(item) > 1:
                    # Bisect
                    n = len(item) // 2
                    left = item[:n]
                    right = item[n:]

                    # Calculate variance for each subset
                    var_left = self._get_cluster_var(cov, left)
                    var_right = self._get_cluster_var(cov, right)

                    # Allocate weights inversely proportional to variance
                    total_var = var_left + var_right
                    if total_var > 0:
                        alpha = var_right / total_var
                    else:
                        alpha = 0.5

                    # Update weights
                    weights.iloc[left] *= alpha
                    weights.iloc[right] *= (1 - alpha)

                    # Add to items
                    items.extend([left, right])

            # Normalize
            return weights.values / (weights.sum() + 1e-8)

        except:
            # Equal weights as fallback
            return np.ones(len(cov)) / len(cov)

    def _get_cluster_var(self, cov: pd.DataFrame, items: List[int]) -> float:
        """Calculate cluster variance"""
        try:
            if len(items) == 0:
                return 0
            elif len(items) == 1:
                return cov.iloc[items[0], items[0]]
            else:
                # Calculate weighted variance
                cluster_cov = cov.iloc[items, items]
                weights = pd.Series(1 / len(items), index=cluster_cov.index)

                return weights @ cluster_cov @ weights
        except:
            return 1.0

class RegimeAwarePortfolioOptimizer:
    """Portfolio optimizer that adapts to market regimes"""

    def __init__(self):
        self.hrp = HierarchicalRiskParity()
        self.regime_weights = {
            'trending_up': {'momentum': 0.6, 'mean_reversion': 0.1,
                           'volatility': 0.1, 'alternative': 0.2},
            'trending_down': {'momentum': 0.2, 'mean_reversion': 0.3,
                            'volatility': 0.3, 'alternative': 0.2},
            'mean_reverting': {'momentum': 0.1, 'mean_reversion': 0.6,
                              'volatility': 0.1, 'alternative': 0.2},
            'volatile': {'momentum': 0.2, 'mean_reversion': 0.2,
                        'volatility': 0.4, 'alternative': 0.2}
        }

    def optimize_portfolio(self, factors: List[AlphaFactor],
                          factor_returns: pd.DataFrame,
                          regime: MarketRegime) -> Dict[str, float]:
        """Optimize portfolio weights based on regime"""

        # Handle empty cases
        if not factors or factor_returns.empty:
            return {}

        # Get regime-specific category weights
        category_weights = self.regime_weights.get(
            regime.regime_type,
            self.regime_weights['volatile']
        )

        # Group factors by category
        factors_by_category = defaultdict(list)
        for factor in factors:
            category = factor.category if factor.category in category_weights else 'alternative'
            factors_by_category[category].append(factor)

        # Calculate weights within each category using HRP
        final_weights = {}

        for category, cat_factors in factors_by_category.items():
            if not cat_factors:
                continue

            # Get returns for factors in this category
            cat_factor_names = [f.name for f in cat_factors]
            available_factors = [name for name in cat_factor_names if name in factor_returns.columns]

            if not available_factors:
                continue

            cat_returns = factor_returns[available_factors]

            if len(cat_returns.columns) == 1:
                # Single factor in category
                within_cat_weights = pd.Series(1.0, index=cat_returns.columns)
            else:
                # Multiple factors - use HRP
                within_cat_weights = self.hrp.calculate_weights(
                    cat_returns,
                    pd.DataFrame()  # No additional scores needed
                )

            # Apply category weight
            cat_weight = category_weights.get(category, 0.1)

            for factor_name, weight in within_cat_weights.items():
                final_weights[factor_name] = weight * cat_weight

        # Normalize weights
        total_weight = sum(final_weights.values())
        if total_weight > 0:
            final_weights = {k: v/total_weight for k, v in final_weights.items()}

        return final_weights

class AlphaMiningPlatform:
    """Main platform for alpha factor discovery and evaluation"""

    def __init__(self, openai_api_key: str = None):
        # Initialize components with API key
        self.llm_generator = LLMAlphaGenerator(api_key=openai_api_key)
        self.alt_data_pipeline = AlternativeDataPipeline()
        self.regime_detector = MarketRegimeDetector()
        self.factor_evaluator = FactorEvaluator()
        self.portfolio_optimizer = RegimeAwarePortfolioOptimizer()

        # Factor storage
        self.discovered_factors = []
        self.active_factors = []
        self.factor_performance_history = defaultdict(list)

        # Portfolio state
        self.current_weights = {}
        self.portfolio_value = 100000
        self.portfolio_history = []

        # Store factor values for backtesting
        self.factor_values_cache = {}

    def discover_factors(self, market_data: pd.DataFrame,
                        n_factors: int = 20) -> List[AlphaFactor]:
        """Discover new alpha factors using multiple methods"""

        discovered = []

        # Get market context for LLM
        current_regime = self.regime_detector.detect_regime(market_data)
        market_context = {
            'regime': current_regime.regime_type,
            'volatility': current_regime.characteristics['volatility'],
            'trend_strength': current_regime.characteristics['trend_strength']
        }

        # 1. Classical factors
        classical_funcs = ClassicalAlphaFactors.get_all_classical_factors()
        for i, func in enumerate(classical_funcs[:n_factors//2]):
            factor = AlphaFactor(
                name=f"classical_{func.__name__}",
                formula=func,
                category="price",
                lookback_period=20
            )
            discovered.append(factor)

        # 2. LLM-generated factors
        categories = ['momentum', 'mean_reversion', 'volatility', 'microstructure']
        for i in range(n_factors//3):
            category = categories[i % len(categories)]
            name, formula = self.llm_generator.generate_llm_factor(
                market_context=market_context,
                category=category
            )

            factor = AlphaFactor(
                name=name,
                formula=formula,
                category=category,
                lookback_period=random.choice([10, 20, 30, 60])
            )
            discovered.append(factor)

        # 3. Sentiment-based factors
        sentiment_scores = self.alt_data_pipeline.extract_sentiment_scores()
        sentiment_factors = self.alt_data_pipeline.create_sentiment_alpha_factors(
            sentiment_scores
        )
        discovered.extend(sentiment_factors[:n_factors//6])

        return discovered

    def evaluate_factors(self, factors: List[AlphaFactor],
                        market_data: pd.DataFrame) -> pd.DataFrame:
        """Evaluate all factors and return performance metrics"""

        # Detect current regime
        regime = self.regime_detector.detect_regime(market_data)

        evaluation_results = []

        # Clear cache for new evaluation
        self.factor_values_cache = {}

        for factor in factors:
            # Evaluate performance
            metrics = self.factor_evaluator.evaluate_factor_performance(
                factor, market_data, regime
            )

            # Update factor attributes
            factor.ic_score = metrics['ic']
            factor.sharpe_ratio = metrics['sharpe_ratio']
            factor.turnover = metrics['turnover']

            # Calculate decay
            decay_profile = self.factor_evaluator.calculate_factor_decay(
                factor, market_data
            )

            # Average decay rate
            if len(decay_profile) > 1:
                decay_values = list(decay_profile.values())
                factor.decay_rate = (decay_values[0] - decay_values[-1]) / len(decay_values)

            # Store regime performance
            factor.regime_performance[regime.regime_type] = metrics['ic']

            # Cache factor values for backtesting
            self.factor_values_cache[factor.name] = self.factor_evaluator._get_factor_values(factor, market_data)

            evaluation_results.append({
                'name': factor.name,
                'category': factor.category,
                'ic': metrics['ic'],
                'sharpe': metrics['sharpe_ratio'],
                'turnover': metrics['turnover'],
                'max_dd': metrics['max_drawdown'],
                'regime': regime.regime_type,
                'decay_rate': factor.decay_rate
            })

        return pd.DataFrame(evaluation_results)

    def select_active_factors(self, factors: List[AlphaFactor],
                             min_ic: float = MIN_FACTOR_IC,
                             max_correlation: float = MAX_FACTOR_CORRELATION) -> List[AlphaFactor]:
        """Select factors for active trading"""

        # Filter by minimum IC
        qualified_factors = [f for f in factors if abs(f.ic_score) > min_ic]

        if not qualified_factors:
            return []

        # Sort by IC
        qualified_factors.sort(key=lambda x: abs(x.ic_score), reverse=True)

        # Select uncorrelated factors
        selected = [qualified_factors[0]]

        for factor in qualified_factors[1:]:
            # Check correlation with selected factors
            correlated = False

            # Calculate actual correlation if we have cached values
            if factor.name in self.factor_values_cache:
                for selected_factor in selected:
                    if selected_factor.name in self.factor_values_cache:
                        corr = self.factor_values_cache[factor.name].corr(
                            self.factor_values_cache[selected_factor.name]
                        )
                        if abs(corr) > max_correlation:
                            correlated = True
                            break
            else:
                # Fallback: assume high correlation within same category
                for selected_factor in selected:
                    if factor.category == selected_factor.category:
                        correlated = True
                        break

            if not correlated:
                selected.append(factor)

            if len(selected) >= 10:  # Maximum active factors
                break

        return selected

    def construct_portfolio(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Construct portfolio based on active factors"""

        # Get current regime
        regime = self.regime_detector.detect_regime(market_data)

        # Generate factor returns for optimization
        factor_returns = pd.DataFrame()

        for factor in self.active_factors:
            # Use actual factor values if available
            if factor.name in self.factor_values_cache:
                factor_values = self.factor_values_cache[factor.name]
                # Calculate factor returns
                ranks = factor_values.rank(pct=True, na_option='keep').fillna(0.5)
                long_weight = (ranks > 0.8).astype(float)
                short_weight = (ranks < 0.2).astype(float)

                # Normalize
                long_sum = long_weight.sum()
                short_sum = short_weight.sum()

                if long_sum > 0:
                    long_weight = long_weight / long_sum
                if short_sum > 0:
                    short_weight = short_weight / short_sum

                # Get market returns
                market_returns = market_data['close'].pct_change()

                # Factor portfolio returns
                factor_return = (long_weight - short_weight) * market_returns
                factor_returns[factor.name] = factor_return

        # Optimize weights
        if not factor_returns.empty and len(factor_returns) > 252:
            weights = self.portfolio_optimizer.optimize_portfolio(
                self.active_factors,
                factor_returns.iloc[-252:],  # Last year
                regime
            )
        else:
            # Equal weights if insufficient data
            weights = {f.name: 1.0/len(self.active_factors) for f in self.active_factors}

        self.current_weights = weights

        # Count categories
        category_counts = defaultdict(int)
        for f in self.active_factors:
            category_counts[f.category] += 1

        return {
            'weights': weights,
            'regime': regime.regime_type,
            'n_factors': len(weights),
            'categories': dict(category_counts)
        }

    def backtest_portfolio(self, market_data: pd.DataFrame,
                          initial_capital: float,
                          rebalance_freq: int) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """Run realistic portfolio backtest"""

        portfolio_value = initial_capital
        portfolio_history = []
        positions = {}

        # Run backtest from day 100 to allow for lookback
        start_idx = min(100, len(market_data) // 3)

        for i in range(start_idx, len(market_data), 1):
            current_date = market_data.index[i]

            # Rebalance if needed
            if i % rebalance_freq == 0 or i == start_idx:
                # Get market data up to current point
                current_data = market_data.iloc[:i]

                # Rebalance portfolio
                portfolio_info = self.construct_portfolio(current_data)

                # Update positions based on new weights
                new_positions = {}
                for factor_name, weight in portfolio_info['weights'].items():
                    new_positions[factor_name] = portfolio_value * weight

                # Calculate transaction costs
                transaction_cost = 0
                for factor_name in set(list(positions.keys()) + list(new_positions.keys())):
                    old_value = positions.get(factor_name, 0)
                    new_value = new_positions.get(factor_name, 0)
                    transaction_cost += abs(new_value - old_value) * TRANSACTION_COST

                portfolio_value -= transaction_cost
                positions = new_positions

            # Calculate daily returns for each factor
            daily_pnl = 0

            for factor_name, position_value in positions.items():
                # Find the factor
                factor = next((f for f in self.active_factors if f.name == factor_name), None)

                if factor and factor_name in self.factor_values_cache:
                    # Get factor value for today
                    factor_values = self.factor_values_cache[factor_name]

                    if i < len(factor_values):
                        # Calculate factor portfolio return for today
                        ranks = factor_values.iloc[:i].rank(pct=True, na_option='keep').fillna(0.5)
                        if len(ranks) > 0:
                            current_rank = ranks.iloc[-1]

                            # Determine position direction
                            if current_rank > 0.8:
                                position_direction = 1
                            elif current_rank < 0.2:
                                position_direction = -1
                            else:
                                position_direction = 0

                            # Today's market return
                            if i > 0:
                                market_return = (market_data['close'].iloc[i] - market_data['close'].iloc[i-1]) / market_data['close'].iloc[i-1]
                            else:
                                market_return = 0

                            # Factor PnL
                            factor_pnl = position_value * position_direction * market_return
                            daily_pnl += factor_pnl

            # Update portfolio value
            portfolio_value += daily_pnl

            # Record history
            portfolio_history.append({
                'date': current_date,
                'value': portfolio_value,
                'pnl': daily_pnl,
                'positions': positions.copy()
            })

        # Convert to DataFrame
        history_df = pd.DataFrame(portfolio_history)

        if history_df.empty:
            return history_df, {
                'total_return': 0.0,
                'annual_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.5
            }

        # Calculate performance metrics
        returns = history_df['pnl'] / history_df['value'].shift(1)
        returns = returns.fillna(0)

        total_return = (portfolio_value - initial_capital) / initial_capital
        annual_return = (portfolio_value / initial_capital) ** (252 / len(history_df)) - 1 if len(history_df) > 0 else 0

        if returns.std() > 0:
            sharpe = np.sqrt(252) * returns.mean() / returns.std()
        else:
            sharpe = 0

        # Max drawdown
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = (running_max - cum_returns) / running_max
        max_drawdown = drawdown.max()

        metrics = {
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': (returns > 0).mean()
        }

        return history_df, metrics

    def calculate_information_coefficient_decay(self,
                                              factor: AlphaFactor,
                                              market_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate and visualize IC decay"""

        decay_profile = self.factor_evaluator.calculate_factor_decay(
            factor, market_data, max_lag=30
        )

        decay_df = pd.DataFrame([
            {'lag': lag, 'ic': ic}
            for lag, ic in decay_profile.items()
        ])

        return decay_df

# Market data generator
class MarketDataGenerator:
    """Generate realistic market data for demonstration"""

    @staticmethod
    def generate_market_data(n_days: int = 1000) -> pd.DataFrame:
        """Generate OHLCV market data"""

        dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')

        # Base price movement
        returns = np.random.normal(0.0005, 0.02, n_days)

        # Add regime changes
        regime_changes = [0, n_days//4, n_days//2, 3*n_days//4, n_days]

        for i in range(len(regime_changes)-1):
            start, end = regime_changes[i], regime_changes[i+1]

            if i % 4 == 0:  # Trending up
                returns[start:end] += np.random.normal(0.001, 0.001, end-start)
            elif i % 4 == 1:  # Mean reverting
                returns[start:end] = np.random.normal(0, 0.015, end-start)
            elif i % 4 == 2:  # Trending down
                returns[start:end] += np.random.normal(-0.001, 0.001, end-start)
            else:  # Volatile
                returns[start:end] = np.random.normal(0, 0.03, end-start)

        # Generate prices
        prices = 100 * np.exp(np.cumsum(returns))

        # Generate OHLCV
        data = pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.001, n_days)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.005, n_days))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.005, n_days))),
            'close': prices,
            'volume': np.random.lognormal(15, 0.5, n_days)
        }, index=dates)

        # Ensure OHLC consistency
        data['high'] = data[['open', 'high', 'close']].max(axis=1)
        data['low'] = data[['open', 'low', 'close']].min(axis=1)

        return data

# Visualization and Gradio Interface
def create_gradio_interface():
    """Create the main Gradio interface for the Alpha Mining Platform"""

    # Initialize the platform
    platform = None
    market_data_cache = {}

    def generate_and_evaluate_factors(n_days, n_factors, min_ic_threshold, openai_api_key):
        """Main function to generate and evaluate alpha factors"""

        try:
            # Initialize platform with API key
            nonlocal platform
            platform = AlphaMiningPlatform(openai_api_key=openai_api_key if openai_api_key else None)

            # Generate market data
            market_data = MarketDataGenerator.generate_market_data(int(n_days))
            market_data_cache['data'] = market_data

            # Discover factors
            discovered_factors = platform.discover_factors(market_data, int(n_factors))
            platform.discovered_factors = discovered_factors

            # Evaluate factors
            evaluation_df = platform.evaluate_factors(discovered_factors, market_data)

            # Select active factors
            platform.active_factors = platform.select_active_factors(
                discovered_factors,
                min_ic=float(min_ic_threshold)
            )

            # Construct portfolio
            portfolio_info = platform.construct_portfolio(market_data)

            # Create visualizations
            # 1. Factor Performance Heatmap
            fig_heatmap = create_factor_heatmap(evaluation_df)

            # 2. IC Distribution
            fig_ic_dist = create_ic_distribution(evaluation_df)

            # 3. Portfolio Weights
            fig_weights = create_portfolio_weights_chart(portfolio_info['weights'])

            # 4. Regime Timeline
            fig_regime = create_regime_timeline(market_data, platform.regime_detector)

            # Create summary statistics
            active_factor_names = [f.name for f in platform.active_factors]
            active_factors_df = evaluation_df[evaluation_df['name'].isin(active_factor_names)]
            avg_ic = active_factors_df['ic'].mean() if len(active_factors_df) > 0 else 0

            summary_stats = f"""
            ### Factor Discovery Summary
            - **Total Factors Discovered**: {len(discovered_factors)}
            - **Active Factors Selected**: {len(platform.active_factors)}
            - **Current Market Regime**: {portfolio_info['regime']}
            - **Average IC of Active Factors**: {avg_ic:.4f}
            - **Average Sharpe Ratio**: {evaluation_df['sharpe'].mean():.2f}

            ### Portfolio Construction
            - **Number of Factors in Portfolio**: {portfolio_info['n_factors']}
            - **Category Distribution**: {portfolio_info['categories']}

            ### LLM-Generated Factors
            - **Total LLM Factors**: {len([f for f in discovered_factors if 'LLM' in f.name or 'Fallback' in f.name])}
            - **LLM Factors Selected**: {len([f for f in platform.active_factors if 'LLM' in f.name or 'Fallback' in f.name])}
            """

            # Top factors table
            top_factors_df = evaluation_df.nlargest(10, 'ic')[
                ['name', 'category', 'ic', 'sharpe', 'turnover', 'regime']
            ].round(3)

            return fig_heatmap, fig_ic_dist, fig_weights, fig_regime, summary_stats, top_factors_df

        except Exception as e:
            print(f"Error in generate_and_evaluate_factors: {e}")
            # Return empty figures if error occurs
            empty_fig = go.Figure()
            empty_fig.add_annotation(text="Error generating data", x=0.5, y=0.5, showarrow=False)
            return empty_fig, empty_fig, empty_fig, empty_fig, f"Error: {str(e)}", pd.DataFrame()

    def analyze_factor_decay(factor_name):
        """Analyze IC decay for a specific factor"""

        try:
            if 'data' not in market_data_cache or platform is None:
                empty_fig = go.Figure()
                empty_fig.add_annotation(text="Please generate factors first", x=0.5, y=0.5, showarrow=False)
                return empty_fig, "Please generate factors first"

            market_data = market_data_cache['data']

            # Find factor
            factor = None
            for f in platform.discovered_factors:
                if f.name == factor_name:
                    factor = f
                    break

            if not factor:
                empty_fig = go.Figure()
                empty_fig.add_annotation(text=f"Factor '{factor_name}' not found", x=0.5, y=0.5, showarrow=False)
                return empty_fig, f"Factor '{factor_name}' not found"

            # Calculate decay
            decay_df = platform.calculate_information_coefficient_decay(factor, market_data)

            # Create decay plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=decay_df['lag'],
                y=decay_df['ic'],
                mode='lines+markers',
                name='IC Decay',
                line=dict(color='blue', width=2),
                marker=dict(size=8)
            ))

            # Add exponential fit
            if len(decay_df) > 3:
                from scipy.optimize import curve_fit

                def exp_decay(x, a, b):
                    return a * np.exp(-b * x)

                try:
                    popt, _ = curve_fit(exp_decay, decay_df['lag'], np.abs(decay_df['ic']))
                    fit_y = exp_decay(decay_df['lag'], *popt)

                    fig.add_trace(go.Scatter(
                        x=decay_df['lag'],
                        y=fit_y,
                        mode='lines',
                        name='Exponential Fit',
                        line=dict(color='red', width=2, dash='dash')
                    ))

                    half_life = np.log(2) / popt[1] if popt[1] > 0 else np.inf
                    decay_stats = f"Half-life: {half_life:.1f} days"
                except:
                    decay_stats = "Could not fit exponential decay"
            else:
                decay_stats = "Insufficient data for decay analysis"

            fig.update_layout(
                title=f"Information Coefficient Decay: {factor_name}",
                xaxis_title="Prediction Horizon (days)",
                yaxis_title="Information Coefficient",
                height=400
            )

            return fig, decay_stats

        except Exception as e:
            print(f"Error in analyze_factor_decay: {e}")
            empty_fig = go.Figure()
            empty_fig.add_annotation(text=f"Error: {str(e)}", x=0.5, y=0.5, showarrow=False)
            return empty_fig, f"Error: {str(e)}"

    def backtest_portfolio(initial_capital, rebalance_freq):
        """Run portfolio backtest with actual factor returns"""

        try:
            if 'data' not in market_data_cache or platform is None or not platform.active_factors:
                empty_fig = go.Figure()
                empty_fig.add_annotation(text="Please generate and select factors first", x=0.5, y=0.5, showarrow=False)
                return empty_fig, "Please generate and select factors first", ""

            market_data = market_data_cache['data']
            initial_capital = float(initial_capital)
            rebalance_freq = int(rebalance_freq)

            # Run realistic backtest
            history_df, metrics = platform.backtest_portfolio(
                market_data, initial_capital, rebalance_freq
            )

            if history_df.empty:
                empty_fig = go.Figure()
                empty_fig.add_annotation(text="No backtest data generated", x=0.5, y=0.5, showarrow=False)
                return empty_fig, "No backtest data generated", ""

            # Create performance chart
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Portfolio Value', 'Rolling Sharpe Ratio'),
                row_heights=[0.7, 0.3],
                vertical_spacing=0.1
            )

            # Portfolio value
            fig.add_trace(
                go.Scatter(
                    x=history_df['date'],
                    y=history_df['value'],
                    mode='lines',
                    name='Portfolio Value',
                    line=dict(color='blue', width=2)
                ),
                row=1, col=1
            )

            # Benchmark (buy and hold)
            market_returns = market_data['close'].pct_change().fillna(0)
            benchmark_value = initial_capital * (1 + market_returns).cumprod()
            benchmark_dates = market_data.index[market_data.index.isin(history_df['date'])]
            benchmark_value = benchmark_value[benchmark_dates]

            fig.add_trace(
                go.Scatter(
                    x=benchmark_dates,
                    y=benchmark_value,
                    mode='lines',
                    name='Buy & Hold Benchmark',
                    line=dict(color='gray', width=1, dash='dash')
                ),
                row=1, col=1
            )

            # Rolling Sharpe
            returns = history_df['pnl'] / history_df['value'].shift(1)
            returns = returns.fillna(0)

            if len(returns) > 60:
                rolling_returns = returns.rolling(window=60)
                rolling_sharpe = np.sqrt(252) * rolling_returns.mean() / (rolling_returns.std() + 1e-8)

                fig.add_trace(
                    go.Scatter(
                        x=history_df['date'],
                        y=rolling_sharpe,
                        mode='lines',
                        name='60-Day Sharpe',
                        line=dict(color='green', width=2)
                    ),
                    row=2, col=1
                )

            fig.update_layout(height=700, showlegend=True)
            fig.update_xaxes(title_text="Date", row=2, col=1)
            fig.update_yaxes(title_text="Portfolio Value", row=1, col=1)
            fig.update_yaxes(title_text="Sharpe Ratio", row=2, col=1)

            # Create metrics text
            metrics_text = f"""
            ### Backtest Performance Metrics
            - **Total Return**: {metrics['total_return']*100:.2f}%
            - **Annualized Return**: {metrics['annual_return']*100:.2f}%
            - **Sharpe Ratio**: {metrics['sharpe_ratio']:.2f}
            - **Maximum Drawdown**: {metrics['max_drawdown']*100:.2f}%
            - **Win Rate**: {metrics['win_rate']*100:.1f}%
            - **Number of Rebalances**: {len(history_df) // rebalance_freq}

            ### Active Factors Used
            {', '.join([f.name for f in platform.active_factors])}
            """

            return fig, metrics_text, ""

        except Exception as e:
            print(f"Error in backtest_portfolio: {e}")
            empty_fig = go.Figure()
            empty_fig.add_annotation(text=f"Error: {str(e)}", x=0.5, y=0.5, showarrow=False)
            return empty_fig, f"Error: {str(e)}", ""

    # Helper visualization functions
    def create_factor_heatmap(eval_df):
        """Create heatmap of factor performance by category"""
        try:
            if eval_df.empty:
                fig = go.Figure()
                fig.add_annotation(text="No data to display", x=0.5, y=0.5, showarrow=False)
                return fig

            # Create pivot table
            pivot_df = pd.pivot_table(
                eval_df,
                values='ic',
                index='category',
                columns='regime',
                aggfunc='mean',
                fill_value=0
            )

            if pivot_df.empty:
                fig = go.Figure()
                fig.add_annotation(text="No data to display", x=0.5, y=0.5, showarrow=False)
                return fig

            fig = go.Figure(data=go.Heatmap(
                z=pivot_df.values,
                x=pivot_df.columns,
                y=pivot_df.index,
                colorscale='RdBu',
                zmid=0,
                text=np.round(pivot_df.values, 3),
                texttemplate='%{text}',
                textfont={"size": 10}
            ))

            fig.update_layout(
                title="Average IC by Factor Category and Market Regime",
                xaxis_title="Market Regime",
                yaxis_title="Factor Category",
                height=400
            )

            return fig
        except Exception as e:
            print(f"Error in create_factor_heatmap: {e}")
            fig = go.Figure()
            fig.add_annotation(text=f"Error: {str(e)}", x=0.5, y=0.5, showarrow=False)
            return fig

    def create_ic_distribution(eval_df):
        """Create IC distribution plot"""
        try:
            if eval_df.empty:
                fig = go.Figure()
                fig.add_annotation(text="No data to display", x=0.5, y=0.5, showarrow=False)
                return fig

            fig = go.Figure()

            for category in eval_df['category'].unique():
                cat_data = eval_df[eval_df['category'] == category]

                fig.add_trace(go.Box(
                    y=cat_data['ic'],
                    name=category,
                    boxpoints='all',
                    jitter=0.3,
                    pointpos=-1.8
                ))

            fig.update_layout(
                title="Information Coefficient Distribution by Category",
                yaxis_title="Information Coefficient",
                showlegend=False,
                height=400
            )

            # Add reference line at 0
            fig.add_hline(y=0, line_dash="dash", line_color="gray")

            return fig
        except Exception as e:
            print(f"Error in create_ic_distribution: {e}")
            fig = go.Figure()
            fig.add_annotation(text=f"Error: {str(e)}", x=0.5, y=0.5, showarrow=False)
            return fig

    def create_portfolio_weights_chart(weights):
        """Create portfolio weights pie chart"""
        try:
            if not weights:
                fig = go.Figure()
                fig.add_annotation(
                    text="No active factors selected",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5,
                    showarrow=False
                )
                fig.update_layout(height=400)
                return fig

            fig = go.Figure(data=[go.Pie(
                labels=list(weights.keys()),
                values=list(weights.values()),
                textposition='inside',
                textinfo='percent+label',
                hole=0.3
            )])

            fig.update_layout(
                title="Portfolio Factor Weights",
                height=400
            )

            return fig
        except Exception as e:
            print(f"Error in create_portfolio_weights_chart: {e}")
            fig = go.Figure()
            fig.add_annotation(text=f"Error: {str(e)}", x=0.5, y=0.5, showarrow=False)
            return fig

    def create_regime_timeline(market_data, regime_detector):
        """Create market regime timeline"""
        try:
            # Detect regimes at different points
            regime_history = []

            step = max(20, len(market_data) // 50)
            for i in range(60, len(market_data), step):
                regime = regime_detector.detect_regime(market_data.iloc[:i])
                regime_history.append({
                    'date': market_data.index[i-1],
                    'regime': regime.regime_type,
                    'confidence': regime.confidence
                })

            regime_df = pd.DataFrame(regime_history)

            if regime_df.empty:
                fig = go.Figure()
                fig.add_annotation(text="No regime data", x=0.5, y=0.5, showarrow=False)
                return fig

            # Create color map
            color_map = {
                'trending_up': 'green',
                'trending_down': 'red',
                'mean_reverting': 'blue',
                'volatile': 'orange'
            }

            fig = go.Figure()

            # Add regime bars
            for regime in color_map.keys():
                regime_data = regime_df[regime_df['regime'] == regime]

                if len(regime_data) > 0:
                    fig.add_trace(go.Scatter(
                        x=regime_data['date'],
                        y=regime_data['confidence'],
                        mode='markers',
                        name=regime,
                        marker=dict(
                            color=color_map[regime],
                            size=10,
                            symbol='square'
                        )
                    ))

            fig.update_layout(
                title="Market Regime Detection Timeline",
                xaxis_title="Date",
                yaxis_title="Confidence",
                height=300,
                yaxis_range=[0, 1]
            )

            return fig
        except Exception as e:
            print(f"Error in create_regime_timeline: {e}")
            fig = go.Figure()
            fig.add_annotation(text=f"Error: {str(e)}", x=0.5, y=0.5, showarrow=False)
            return fig

    # Create Gradio interface
    with gr.Blocks(title="Quantitative Alpha Mining Platform") as interface:
        gr.Markdown("""
        # Quantitative Alpha Mining Platform with LLM Discovery

        This platform leverages LLMs and machine learning to discover novel alpha factors from multiple data sources:
        - **Classical Factors**: Implementation of quantitative factors inspired by WorldQuant's research
        - **LLM-Generated Factors**: Novel factor formulas created using OpenAI's GPT models
        - **Alternative Data**: Sentiment analysis from earnings calls, SEC filings, news, and social media
        - **Regime-Aware Portfolio**: Hierarchical Risk Parity with dynamic regime adaptation

        Author: Spencer Purdy
        """)

        with gr.Tab("Factor Discovery"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Configuration")
                    n_days = gr.Slider(
                        minimum=500, maximum=2000, value=1000, step=100,
                        label="Market Data Days"
                    )
                    n_factors = gr.Slider(
                        minimum=10, maximum=50, value=20, step=5,
                        label="Number of Factors to Generate"
                    )
                    min_ic = gr.Slider(
                        minimum=0.01, maximum=0.1, value=0.02, step=0.01,
                        label="Minimum IC Threshold"
                    )

                    gr.Markdown("### API Configuration")
                    openai_api_key = gr.Textbox(
                        label="OpenAI API Key",
                        placeholder="sk-...",
                        type="password",
                        info="Optional: For LLM-generated factors (leave empty for fallback)"
                    )

                    generate_btn = gr.Button("Generate & Evaluate Factors", variant="primary")

            with gr.Row():
                factor_heatmap = gr.Plot(label="Factor Performance Heatmap")
                ic_distribution = gr.Plot(label="IC Distribution")

            with gr.Row():
                portfolio_weights = gr.Plot(label="Portfolio Weights")
                regime_timeline = gr.Plot(label="Market Regime Timeline")

            with gr.Row():
                summary_stats = gr.Markdown(label="Summary Statistics")
                top_factors_table = gr.DataFrame(label="Top Factors by IC")

        with gr.Tab("Factor Analysis"):
            with gr.Row():
                with gr.Column(scale=1):
                    factor_selector = gr.Dropdown(
                        choices=[],
                        label="Select Factor to Analyze"
                    )
                    analyze_btn = gr.Button("Analyze Factor Decay")

                with gr.Column(scale=2):
                    decay_plot = gr.Plot(label="IC Decay Analysis")
                    decay_stats = gr.Markdown(label="Decay Statistics")

        with gr.Tab("Portfolio Backtest"):
            with gr.Row():
                with gr.Column(scale=1):
                    initial_capital_input = gr.Number(
                        value=100000, label="Initial Capital", minimum=10000
                    )
                    rebalance_freq_input = gr.Slider(
                        minimum=5, maximum=60, value=20, step=5,
                        label="Rebalance Frequency (days)"
                    )

                    backtest_btn = gr.Button("Run Backtest", variant="primary")

                with gr.Column(scale=2):
                    backtest_plot = gr.Plot(label="Backtest Performance")
                    backtest_metrics = gr.Markdown(label="Performance Metrics")
                    backtest_error = gr.Markdown(visible=False)

        # Event handlers
        def update_factor_selector(fig1, fig2, fig3, fig4, stats, table):
            """Update factor selector with discovered factors"""
            if platform and platform.discovered_factors:
                choices = [f.name for f in platform.discovered_factors]
                return gr.Dropdown(choices=choices, value=choices[0] if choices else None)
            return gr.Dropdown(choices=[])

        generate_btn.click(
            fn=generate_and_evaluate_factors,
            inputs=[n_days, n_factors, min_ic, openai_api_key],
            outputs=[factor_heatmap, ic_distribution, portfolio_weights,
                    regime_timeline, summary_stats, top_factors_table]
        ).then(
            fn=update_factor_selector,
            inputs=[factor_heatmap, ic_distribution, portfolio_weights,
                   regime_timeline, summary_stats, top_factors_table],
            outputs=[factor_selector]
        )

        analyze_btn.click(
            fn=analyze_factor_decay,
            inputs=[factor_selector],
            outputs=[decay_plot, decay_stats]
        )

        backtest_btn.click(
            fn=backtest_portfolio,
            inputs=[initial_capital_input, rebalance_freq_input],
            outputs=[backtest_plot, backtest_metrics, backtest_error]
        )

        # Add examples
        gr.Examples(
            examples=[
                [1000, 20, 0.02],
                [1500, 30, 0.03],
                [2000, 40, 0.025]
            ],
            inputs=[n_days, n_factors, min_ic]
        )

        gr.Markdown("""
        ---
        **Note**: This system uses sophisticated machine learning models including optional LLM integration for factor discovery.
        For best results, provide an OpenAI API key for genuine LLM-generated factors. Without an API key, the system will use
        fallback factor generation methods. The simulation and analysis features work with or without the API key.
        All trading strategies are for demonstration purposes only.

        **API Key Information**:
        - OpenAI API Key: Get yours at https://platform.openai.com/api-keys
        """)

    return interface

# Launch the application
if __name__ == "__main__":
    interface = create_gradio_interface()
    interface.launch()