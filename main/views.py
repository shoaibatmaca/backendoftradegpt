from django.shortcuts import render
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated, AllowAny
import jwt
from datetime import datetime, timedelta
from django.conf import settings
from rest_framework import status
from .models import ChatSession, ChatMessage
from .utils import get_user_from_token
from django.utils.timezone import now
import uuid
from django.utils import timezone


class TradeGPTUserView(APIView):
    permission_classes = [AllowAny]

    def get(self, request):
        token = request.GET.get("token")
        if not token:
            return Response({"error": "Token is missing"}, status=400)

        try:
            decoded = jwt.decode(token, settings.SECRET_KEY, algorithms=["HS256"])
            return Response({
                "user_id": decoded.get("user_id"),
                "username": decoded.get("username"),
                "first_name": decoded.get("first_name"),
                "last_name": decoded.get("last_name"),
                "email": decoded.get("email"),
                "subscription_status": decoded.get("subscription_status"),
                "profile_photo": decoded.get("profile_photo"),
                "phone_number": decoded.get("phone_number"),
                "country": decoded.get("country"),
                "state": decoded.get("state"),
                "is_staff": decoded.get("is_staff"),
                "is_superuser": decoded.get("is_superuser"),
            })
        except jwt.ExpiredSignatureError:
            return Response({"error": "Token expired"}, status=401)
        except jwt.InvalidTokenError:
            return Response({"error": "Invalid token"}, status=401)


class StartChatSessionView(APIView):
    def get(self, request):
        token = request.GET.get("token")
        user = get_user_from_token(token)

        session = ChatSession.objects.create(
            session_id=uuid.uuid4(),
            user_id=user["user_id"],
            username=user["username"],
        )
        return Response({"session_id": session.session_id})
    
    # Replace POST method with this logic:
    def post(self, request):
        token = request.GET.get("token")
        user = get_user_from_token(token)

        # ✅ Check if session for today already exists
        today = timezone.now().date()
        existing_session = ChatSession.objects.filter(
            user_id=user["user_id"],
            created_at__date=today
        ).first()

        if existing_session:
            return Response({"session_id": existing_session.session_id})
        else:
            session = ChatSession.objects.create(
                session_id=uuid.uuid4(),
                user_id=user["user_id"],
                username=user["username"],
            )
            return Response({"session_id": session.session_id})



    # def post(self, request):
    #     token = request.GET.get("token")
    #     user = get_user_from_token(token)

    #     session = ChatSession.objects.create(
    #         session_id=uuid.uuid4(),
    #         user_id=user["user_id"],
    #         username=user["username"],
    #     )
    #     return Response({"session_id": session.session_id})


class MessageListCreateView(APIView):
    def post(self, request, session_id):
        token = request.GET.get("token")
        user = get_user_from_token(token)

        data = request.data
        ChatMessage.objects.create(
            session_id=session_id,
            role=data["role"],
            content=data["content"]
        )
        return Response({"message": "Saved"}, status=201)

    def get(self, request, session_id):
        token = request.GET.get("token")
        user = get_user_from_token(token)

        messages = ChatMessage.objects.filter(session_id=session_id).order_by("timestamp")
        return Response([
            {"role": m.role, "content": m.content, "timestamp": m.timestamp}
            for m in messages
        ])


class UserChatSessionsView(APIView):
    def get(self, request):
        token = request.GET.get("token")
        user = get_user_from_token(token)

        sessions = ChatSession.objects.filter(user_id=user["user_id"]).order_by("-created_at")
        return Response([
            {"session_id": s.session_id, "created_at": s.created_at}
            for s in sessions
        ])


class DailyMessageLimitView(APIView):
    def get(self, request):
        token = request.GET.get("token")
        user = get_user_from_token(token)

        count = ChatMessage.objects.filter(
            session__user_id=user["user_id"],
            timestamp__date=now().date()
        ).count()

        max_allowed = {
            "free": 3,
            "premium": 5,
            "platinum": 10,
        }.get(user["subscription_status"], 3)

        return Response({"count": count, "max": max_allowed})

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import AllowAny
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from .utils import get_user_from_token
import requests
import re

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import AllowAny
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from .utils import get_user_from_token
import requests



# import re
# import logging

# from django.http import StreamingHttpResponse
# from rest_framework.views import APIView
# from rest_framework.permissions import AllowAny
# from django.views.decorators.csrf import csrf_exempt
# from django.utils.decorators import method_decorator
# from openai import OpenAI
# import time

# logger = logging.getLogger(__name__)


# # def clean_special_chars(text):
# #     import re

# #     # Remove markdown styling (bold, italic, code)
# #     text = re.sub(r'\*\*\*(.*?)\*\*\*', r'\1', text)
# #     text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
# #     text = re.sub(r'\*(.*?)\*', r'\1', text)
# #     text = re.sub(r'`{1,3}(.*?)`{1,3}', r'\1', text)

# #     # Convert markdown headers (## Section) → "Section:"
# #     text = re.sub(r'^#{1,6}\s*(.+)$', r'\1:', text, flags=re.MULTILINE)

# #     # Remove excessive --- or tables like |...|...|
# #     text = re.sub(r'^\|.*?\|$', '', text, flags=re.MULTILINE)  # remove table lines
# #     text = re.sub(r'-{3,}', '\n' + '-'*20 + '\n', text)

# #     # Normalize spacing and line breaks
# #     text = re.sub(r'\n{2,}', '\n\n', text)
# #     text = re.sub(r'\s{2,}', ' ', text)

# #     return text.strip()
# def clean_special_chars(text):
#     import re

#     # Remove markdown styling
#     text = re.sub(r'\*\*\*(.*?)\*\*\*', r'\1', text)  # bold-italic
#     text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)       # bold
#     text = re.sub(r'\*(.*?)\*', r'\1', text)           # italic
#     text = re.sub(r'`{1,3}(.*?)`{1,3}', r'\1', text)   # code

#     # Replace headings (## Heading) with properly formatted section titles
#     text = re.sub(r'^#{1,6}\s*(.+)$', r'\n\n### \1\n', text, flags=re.MULTILINE)

#     # Remove markdown tables and separators
#     text = re.sub(r'\|.*?\|', '', text)         # remove markdown table rows
#     text = re.sub(r'-{3,}', '\n' + '-'*20 + '\n', text)  # normalize separators

#     # Normalize spacing
#     text = re.sub(r'\n{2,}', '\n\n', text)
#     text = re.sub(r'\s{2,}', ' ', text)

#     return text.strip()



# def normalize_query_type(raw):
#     raw = raw.lower().strip()
#     if "price" in raw and "chart" in raw:
#         return "price_chart"
#     elif "news" in raw:
#         return "recent_news"
#     elif "fundamental" in raw or "technical" in raw:
#         return "fundamental_analysis"
#     else:
#         return "default"

# @method_decorator(csrf_exempt, name='dispatch')
# class DeepSeekChatView(APIView):
#     permission_classes = [AllowAny]

#     def post(self, request):
#         try:
#             data = request.data

#             symbol = data.get("symbol", "N/A")
#             name = data.get("name", "N/A")
#             query_type = normalize_query_type(data.get("queryType", "default"))
#             price = data.get("price", "N/A")
#             open_ = data.get("open", "N/A")
#             high = data.get("high", "N/A")
#             low = data.get("low", "N/A")
#             previous_close = data.get("previousClose", "N/A")
#             volume = data.get("volume", "N/A")
#             trend = data.get("trend", "N/A")
#             news_list = data.get("news", [])

#             news_lines = ""
#             for item in news_list[:5]:
#                 headline = item.get("headline", "No headline")
#                 time_str = item.get("time", "Unknown time")
#                 category = item.get("category", "General")
#                 news_lines += f"- {headline} at {time_str} | {category}\n"

#             if not news_lines.strip():
#                 news_lines = "No major headlines available."

#             # Build prompt
#             if query_type == "price_chart":
#                 prompt = f"""
# Act as a financial data analyst. Generate a markdown section showing recent price action for {name} ({symbol}). Include:
# - Volatility patterns
# - Trend direction
# - Notable price movements

# ## Price Movements  
# Price: ${price}, Open: ${open_}, High: ${high}, Low: ${low}, Previous Close: ${previous_close}  
# Volume: {volume}  
# Trend: {trend}
# """
#             elif query_type == "recent_news":
#                 prompt = f"""
# Act as a financial news summarizer. Provide a markdown list of the most recent headlines for {name} ({symbol}). Highlight insights by theme.

# ## Recent News  
# {news_lines}
# """
#             elif query_type == "fundamental_analysis":
#                 prompt = f"""
# Act as an expert financial analyst. Provide a markdown breakdown of {name} ({symbol}).

# ## Company Overview  
# **Symbol:** {symbol}  
# **Company:** {name}  
# **Price:** ${price}  
# **Open:** ${open_}  
# **High:** ${high}  
# **Low:** ${low}  
# **Previous Close:** ${previous_close}  
# **Volume:** {volume}  
# **Trend:** {trend}  

# ## News Headlines  
# {news_lines}

# ## Key Financial Metrics  
# List valuation ratios, margins, ROE, and KPIs.

# ## Strategic Initiatives  
# Mention growth areas or major projects.

# ## Upcoming Events  
# Include earnings dates and financial releases.

# ## Analyst Insights  
# Summarize bullish/bearish sentiment.

# ## Risks  
# Mention major financial or regulatory risks.
# """
#             else:
#                 prompt = f"""
# Act as a professional trader. Based on recent price and news data, suggest a technical trade idea for {name} ({symbol}) including entry, stop-loss, target, and reasoning.

# **Symbol:** {symbol}  
# **Company:** {name}  
# **Price:** ${price}  
# **Open:** ${open_}  
# **High:** ${high}  
# **Low:** ${low}  
# **Previous Close:** ${previous_close}  
# **Volume:** {volume}  
# **Trend:** {trend}  

# ## News Headlines  
# {news_lines}

# ## Trade Setup  
# Explain entry, stop-loss, target and technical indicators.
# """

#             # Streamed Response
#             client = OpenAI(
#                 api_key="sk-fd092005f2f446d78dade7662a13c896",
#                 base_url="https://api.deepseek.com"
#             )

#             response = client.chat.completions.create(
#                 model="deepseek-chat",
#                 messages=[
#                     {"role": "system", "content": "You are TradeGPT, a professional market analyst."},
#                     {"role": "user", "content": prompt}
#                 ],
#                 stream=True
#             )

#             def stream():
#                 for chunk in response:
#                     content = chunk.choices[0].delta.content
#                     if content:
#                         # yield f"data: {content}\n\n"  # Correct SSE format
#                          # Ensure proper line breaks and spacing
#                         # content = content.replace("\n", "\n\n").replace("**", "** ")
#                         content = clean_special_chars(content)

#                         yield f"data: {content}\n\n"
                        

#             return StreamingHttpResponse(stream(), content_type="text/event-stream")


#         except Exception as e:
#             logger.error(f"Streaming error: {str(e)}")
#             return Response({"error": str(e)}, status=500)






import re
import logging
import time

from django.http import StreamingHttpResponse
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import AllowAny
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from openai import OpenAI

logger = logging.getLogger(__name__)


def clean_special_chars(text):
    # Remove markdown styling
    text = re.sub(r'\*\*\*(.*?)\*\*\*', r'\1', text)
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'`{1,3}(.*?)`{1,3}', r'\1', text)
    # Remove markdown headings like ### 1. Tesla (TSLA) → just "1. Tesla (TSLA)"
    text = re.sub(r'^#{1,6}\s*(\d+\.\s?[A-Z].+)', r'\1', text, flags=re.MULTILINE)

# For all other headings, you can still keep optional formatting (or remove this too)
    text = re.sub(r'^#{1,6}\s*(.+)$', r'\n\n\1\n', text, flags=re.MULTILINE)


    # Replace headings (## Heading) with properly formatted section titles
    text = re.sub(r'^#{1,6}\s*(.+)$', r'\n\n### \1\n', text, flags=re.MULTILINE)

    # Remove markdown tables and separators
    text = re.sub(r'\|.*?\|', '', text)  # remove markdown table rows
    text = re.sub(r'-{3,}', '\n' + '-' * 20 + '\n', text)

    # Normalize spacing
    text = re.sub(r'\n{2,}', '\n\n', text)
    text = re.sub(r'\s{2,}', ' ', text)

    return text.strip()


def normalize_query_type(raw):
    raw = raw.lower().strip()
    if "price" in raw and "chart" in raw:
        return "price_chart"
    elif "news" in raw:
        return "recent_news"
    elif "fundamental" in raw or "technical" in raw:
        return "fundamental_analysis"
    else:
        return "default"


@method_decorator(csrf_exempt, name='dispatch')
class DeepSeekChatView(APIView):
    permission_classes = [AllowAny]
    

    def post(self, request):
        try:
            data = request.data

            symbol = data.get("symbol", "N/A")
            name = data.get("name", "N/A")
            query_type = normalize_query_type(data.get("queryType", "default"))
            price = data.get("price", "N/A")
            open_ = data.get("open", "N/A")
            high = data.get("high", "N/A")
            low = data.get("low", "N/A")
            previous_close = data.get("previousClose", "N/A")
            volume = data.get("volume", "N/A")
            trend = data.get("trend", "N/A")
            news_list = data.get("news", [])

            MAX_TOKENS = min(max(int(data.get("tokenLimit", 1500)), 1), 8192)
            news_lines = ""
            for item in news_list[:5]:
                headline = item.get("headline", "No headline")
                time_str = item.get("time", "Unknown time")
                category = item.get("category", "General")
                news_lines += f"- {headline} at {time_str} | {category}\n"

            if not news_lines.strip():
                news_lines = "No major headlines available."

            # Build prompt
            if query_type == "price_chart":
                prompt = f"""
Act as a financial data analyst. Generate a markdown section showing recent price action for {name} ({symbol}). Include:
- Volatility patterns
- Trend direction
- Notable price movements

## Price Movements  
Price: ${price}, Open: ${open_}, High: ${high}, Low: ${low}, Previous Close: ${previous_close}  
Volume: {volume}  
Trend: {trend}
"""
            elif query_type == "recent_news":
                prompt = f"""
Act as a financial news summarizer. Provide a markdown list of the most recent headlines for {name} ({symbol}). Highlight insights by theme.

## Recent News  
{news_lines}
"""
            elif query_type == "fundamental_analysis":
                prompt = f"""
Act as an expert financial analyst. Provide a detailed markdown breakdown of {name} ({symbol}).

## Company Overview  
**Symbol:** {symbol}  
**Company:** {name}  
**Price:** ${price}  
**Open:** ${open_}  
**High:** ${high}  
**Low:** ${low}  
**Previous Close:** ${previous_close}  
**Volume:** {volume}  
**Trend:** {trend}  

## News Headlines  
{news_lines}

## Key Financial Metrics  
List valuation ratios, margins, ROE, and KPIs.

## Strategic Initiatives  
Mention growth areas or major projects.

## Upcoming Events  
Include earnings dates and financial releases.

## Analyst Insights  
Summarize bullish/bearish sentiment.

## Industry Trends  
Discuss broader sector or industry movements that may influence this stock. Include trends such as economic indicators, Fed policies, sector performance, or geopolitical factors. For example:  
- Technology sector resilience  
- Fed interest rate outlook  
- Inflation and consumer demand  
- Global supply chain effects

## Buy and Sell Reasons  
- **Buy:** List technical and fundamental reasons to enter a long trade now.  
- **Sell:** List risks such as weakening earnings, competition, valuation concerns, or macro trends.

## Risks  
Mention major financial or regulatory risks.
"""
#             else:
#                 prompt = f"""
# Act as a professional trader. Based on recent price and news data, suggest a technical trade idea for {name} ({symbol}) including entry, stop-loss, target, and reasoning.

# **Symbol:** {symbol}  
# **Company:** {name}  
# **Price:** ${price}  
# **Open:** ${open_}  
# **High:** ${high}  
# **Low:** ${low}  
# **Previous Close:** ${previous_close}  
# **Volume:** {volume}  
# **Trend:** {trend}  

# ## News Headlines  
# {news_lines}

# ## Trade Setup  
# Explain entry, stop-loss, target and technical indicators.
# """

            else:
                    prompt = f"""
Act as a senior technical analyst and trader. Provide a detailed markdown-based trade ideas valour setup for {name} ({symbol}) based on the latest market data and headlines. Ensure that all sections below are filled with actionable insights.

**Symbol:** {symbol}  
**Company:** {name}  
**Price:** ${price}  
**Open:** ${open_}  
**High:** ${high}  
**Low:** ${low}  
**Previous Close:** ${previous_close}  
**Volume:** {volume}  
**Trend:** {trend}  

## News Headlines  
{news_lines}

## Trade Ideas setup by valourGpt  
Explain entry price, stop-loss, target price, and supporting technical indicators like RSI, MACD, volume trend, support/resistance, or moving averages. Mention any candlestick patterns if relevant.

## Key Financial Metrics (Trailing Twelve Months)  
Include EPS, Gross Margin, Net Margin, Operating Margin, P/E, P/B, P/S, ROA, ROE, Debt/Equity, and Current Ratio. Compare to sector medians if possible.

## Upcoming Events  
Mention scheduled earnings, economic data releases, product launches, or market-moving events that could affect the stock.

## Analyst Insights  
Summarize valuation stance, growth potential, profitability strengths, and momentum. Include any recent earnings revisions or institutional commentary.

## Competitors  
List 2–3 direct competitors. Mention Amazon/Google/Microsoft-type peers and what differentiates this company.

## Unique Value Proposition  
Describe what makes this company valuable long-term — e.g., technology leadership, distribution advantage, IP, customer base, etc.

## Buy and Sell Reasons  
- **Buy:** List technical and fundamental reasons to enter a long trade now.  
- **Sell:** List risks such as weakening earnings, competition, valuation concerns, or macro trends.
"""

            # Initialize client
            client = OpenAI(
                api_key="sk-fd092005f2f446d78dade7662a13c896",
                base_url="https://api.deepseek.com"
            )

            # Generate response (with token limit)
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are TradeGPT, a professional market analyst."},
                    {"role": "user", "content": prompt}
                ],
                stream=True,
                
                max_tokens=MAX_TOKENS
            )

            def stream():
                for chunk in response:
                    content = chunk.choices[0].delta.content
                    if content:
                        yield f"data: {clean_special_chars(content)}\n\n"

            return StreamingHttpResponse(stream(), content_type="text/event-stream")

        except Exception as e:
            logger.error(f"Streaming error: {str(e)}")
            return Response({"error": str(e)}, status=500)


# ===============================================================================
# # direct chat 
@method_decorator(csrf_exempt, name='dispatch')
class DirectChatAIView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        try:
            message = request.data.get("message", "").strip()
            if not message:
                return Response({"error": "Message is required."}, status=400)

            prompt = f"""
You are TradeGPT, a professional market analyst and assistant. Respond clearly in markdown format and provide complete explanations.

User: {message}
"""

            client = OpenAI(
                api_key="sk-fd092005f2f446d78dade7662a13c896",
                base_url="https://api.deepseek.com"
            )

            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are TradeGPT, a helpful financial assistant."},
                    {"role": "user", "content": prompt}
                ],
                stream=True,
                max_tokens=1200
            )

            def stream():
                for chunk in response:
                    content = chunk.choices[0].delta.content
                    if content:
                        yield f"data: {clean_special_chars(content)}\n\n"

            return StreamingHttpResponse(stream(), content_type="text/event-stream")

        except Exception as e:
            logger.error(f"Direct chat error: {str(e)}")
            return Response({"error": str(e)}, status=500)
# ===============================================================================


