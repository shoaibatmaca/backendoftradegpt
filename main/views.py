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

    def post(self, request):
        token = request.GET.get("token")
        user = get_user_from_token(token)

        session = ChatSession.objects.create(
            session_id=uuid.uuid4(),
            user_id=user["user_id"],
            username=user["username"],
        )
        return Response({"session_id": session.session_id})


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

# this is for postman testing with this prompt:

# {
#   "model": "meta-llama/Meta-Llama-3-8B-Instruct",
#   "messages": [
#     { "role": "user", "content": "What are the top 3 call option contracts today?" }
#   ]
# }

# @method_decorator(csrf_exempt, name='dispatch')
# class OpenRouterProxyView(APIView):
#     permission_classes = [AllowAny]

#     def post(self, request):
#         token = request.GET.get("token")
#         if not token:
#             return Response({"error": "Token is missing"}, status=400)

#         try:
#             user = get_user_from_token(token)
#         except Exception as e:
#             return Response({"error": str(e)}, status=401)

#         model = request.data.get("model", "meta-llama/Meta-Llama-3-8B-Instruct")
#         messages = request.data.get("messages", [])

#         if not messages:
#             return Response({"error": "Missing messages"}, status=400)

#         # ✅ Construct prompt using DeepInfra's format
#         prompt = "<|begin_of_text|>"
#         for m in messages:
#             role = m["role"]
#             content = m["content"]
#             prompt += f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
#         prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"

#         payload = {
#             "input": prompt,
#             "stop": ["<|eot_id|>"]
#         }

#         try:
#             res = requests.post(
#                 f"https://api.deepinfra.com/v1/inference/{model}",
#                 headers={
#                     "Authorization": "Bearer FO6ABeaUsSMh82prJuEF2U6uDcBXnBLt",
#                     "Content-Type": "application/json",
#                 },
#                 json=payload,
#                 timeout=60,
#             )
#             res.raise_for_status()
#             data = res.json()

#             return Response({
#                 "message": data["results"][0]["generated_text"]
#             })

#         except requests.exceptions.RequestException as e:
#             return Response({"error": f"DeepInfra request failed: {str(e)}"}, status=500)
#         except Exception as e:
#             return Response({"error": f"Unexpected error: {str(e)}"}, status=500)




# class OpenRouterProxyView(APIView):
#     permission_classes = [AllowAny]

#     def post(self, request):
#         token = request.GET.get("token")
#         if not token:
#             return Response({"error": "Token is missing"}, status=400)

#         try:
#             user = get_user_from_token(token)
#         except Exception as e:
#             return Response({"error": str(e)}, status=401)

#         model = request.data.get("model", "meta-llama/Meta-Llama-3-8B-Instruct")
#         inputs = request.data.get("inputs")  # ✅ NOT `messages`

#         if not inputs or "prompt" not in inputs:
#             return Response({"error": "Missing prompt in 'inputs'"}, status=400)

#         url = f"https://api.deepinfra.com/v1/inference/{model}"

#         try:
#             res = requests.post(
#                 url,
#                 headers={
#                     "Authorization": "Bearer YOUR_DEEPINFRA_API_KEY",
#                     "Content-Type": "application/json",
#                 },
#                 json={
#                     "input": inputs["prompt"],  # ✅ DeepInfra expects `input`, not `inputs`
#                     "stop": ["<|eot_id|>"]
#                 },
#                 timeout=60,
#             )
#             res.raise_for_status()
#             data = res.json()

#             return Response({
#                 "message": data["results"][0]["generated_text"]
#             })

#         except requests.exceptions.RequestException as e:
#             return Response({"error": f"DeepInfra request failed: {str(e)}"}, status=500)
#         except Exception as e:
#             return Response({"error": f"Unexpected error: {str(e)}"}, status=500)

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import AllowAny
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from .utils import get_user_from_token
import requests


@method_decorator(csrf_exempt, name='dispatch')
class OpenRouterProxyView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        token = request.GET.get("token")
        if not token:
            return Response({"error": "Token is missing"}, status=400)

        try:
            user = get_user_from_token(token)
        except Exception as e:
            return Response({"error": str(e)}, status=401)

        model = request.data.get("model")
        inputs = request.data.get("inputs", {})
        prompt = inputs.get("prompt")

        if not prompt:
            return Response({"error": "Missing prompt in 'inputs'"}, status=400)

        url = f"https://api.deepinfra.com/v1/inference/{model}"

        try:
            res = requests.post(
                url,
                headers={
                    "Authorization": "Bearer FO6ABeaUsSMh82prJuEF2U6uDcBXnBLt",
                    "Content-Type": "application/json",
                },
                json={
                    "input": prompt, 
                    "stop": ["<|eot_id|>"]
                },
                timeout=60,
            )
            res.raise_for_status()
            data = res.json()

            return Response({
                "message": data["results"][0]["generated_text"]
            })

        except requests.exceptions.RequestException as e:
            return Response({"error": f"DeepInfra request failed: {str(e)}"}, status=500)
        except Exception as e:
            return Response({"error": f"Unexpected error: {str(e)}"}, status=500)




# from rest_framework.views import APIView
# from rest_framework.response import Response
# from rest_framework.permissions import AllowAny
# from django.views.decorators.csrf import csrf_exempt
# from django.utils.decorators import method_decorator
# import requests


# @method_decorator(csrf_exempt, name='dispatch')
# class DeepSeekChatView(APIView):
#     permission_classes = [AllowAny]

#     def post(self, request):
#         data = request.data

#         symbol = data.get("symbol")
#         name = data.get("name")
#         query_type = data.get("queryType")

#         price = data.get("price", "N/A")
#         open_ = data.get("open", "N/A")
#         high = data.get("high", "N/A")
#         low = data.get("low", "N/A")
#         previous_close = data.get("previousClose", "N/A")
#         volume = data.get("volume", "N/A")
#         trend = data.get("trend", "N/A")
#         news_list = data.get("news", [])

#         news_lines = ""
#         for n in news_list:
#             headline = n.get("headline", "No headline")
#             time = n.get("time", "Unknown time")
#             category = n.get("category", "General")
#             news_lines += f"- **{headline}** at *{time}* | *{category}*\n"

#         prompt = f"""
# Act as an expert financial analyst and return your analysis in clear markdown format.

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
# **Query Type:** {query_type}  

# ## News Headlines  
# {news_lines or '*No major headlines available.*'}

# ## Key Financial Metrics  
# List valuation ratios, margins, ROE, and any known financial KPIs.

# ## Strategic Initiatives  
# Mention growth areas, innovations, or major company projects.

# ## Upcoming Events  
# Include earnings dates, estimates, and any financial releases.

# ## Analyst Insights  
# Summarize bullish/bearish factors, estimates, momentum, and sentiment.

# ## Risks  
# Highlight major financial, regulatory, or competitive risks.

# Respond in this structure with all values you can infer. Format field labels in **bold** for frontend readability.
#         """

#         headers = {
#             "Authorization": "Bearer sk-fd092005f2f446d78dade7662a13c896",
#             "Content-Type": "application/json"
#         }

#         payload = {
#             "model": "deepseek-chat",
#             "messages": [
#                 {"role": "system", "content": "You are TradeGPT, a professional market analyst."},
#                 {"role": "user", "content": prompt}
#             ],
#             "temperature": 0.7,
#             "stream": False
#         }

#         try:
#             res = requests.post("https://api.deepseek.com/v1/chat/completions", headers=headers, json=payload)
#             res.raise_for_status()
#             result = res.json()
#             return Response({"message": result["choices"][0]["message"]["content"]})
#         except Exception as e:
#             return Response({"error": str(e)}, status=500)




# from rest_framework.views import APIView
# from rest_framework.response import Response
# from rest_framework.permissions import AllowAny
# from django.views.decorators.csrf import csrf_exempt
# from django.utils.decorators import method_decorator
# from openai import OpenAI  # DeepSeek uses OpenAI-compatible SDK
# import time

# @method_decorator(csrf_exempt, name='dispatch')
# class DeepSeekChatView(APIView):
#     permission_classes = [AllowAny]

#     def post(self, request):
#         data = request.data

#         symbol = data.get("symbol")
#         name = data.get("name")
#         query_type = data.get("queryType")

#         price = data.get("price", "N/A")
#         open_ = data.get("open", "N/A")
#         high = data.get("high", "N/A")
#         low = data.get("low", "N/A")
#         previous_close = data.get("previousClose", "N/A")
#         volume = data.get("volume", "N/A")
#         trend = data.get("trend", "N/A")
#         news_list = data.get("news", [])

#         news_lines = ""
#         for n in news_list[:5]:
#                 if isinstance(n, dict):
#                     headline = n.get("headline", "No headline")
#                     news_time = n.get("time", "Unknown time")  # renamed to avoid conflict with import
#                     category = n.get("category", "General")
#                     news_lines += f"- **{headline}** at *{news_time}* | *{category}*\n"
#                 else:
#                    news_lines += f"- **{str(n)}**\n"

#         # Construct prompt
#         prompt = f"""
# Act as an expert financial analyst and return your analysis in clear markdown format.

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
# **Query Type:** {query_type}  

# ## News Headlines  
# {news_lines or '*No major headlines available.*'}

# ## Key Financial Metrics  
# List valuation ratios, margins, ROE, and any known financial KPIs.

# ## Strategic Initiatives  
# Mention growth areas, innovations, or major company projects.

# ## Upcoming Events  
# Include earnings dates, estimates, and any financial releases.

# ## Analyst Insights  
# Summarize bullish/bearish factors, estimates, momentum, and sentiment.

# ## Risks  
# Highlight major financial, regulatory, or competitive risks.

# Respond in this structure with all values you can infer. Format field labels in **bold** for frontend readability.
# """

#         try:
#             print("Calling DeepSeek API...")
#             start = time.time()

#             client = OpenAI(
#                 api_key="sk-fd092005f2f446d78dade7662a13c896",  # Make sure this is valid
#                 base_url="https://api.deepseek.com"  # or https://api.deepseek.com/v1
#             )

#             response = client.chat.completions.create(
#                 model="deepseek-chat",  
#                 messages=[
#                     {"role": "system", "content": "You are TradeGPT, a professional market analyst."},
#                     {"role": "user", "content": prompt},
#                 ],
#                 stream=False,
#                 timeout=50
#             )

#             end = time.time()
#             print(f"DeepSeek response time: {end - start:.2f} seconds")

#             return Response({"message": response.choices[0].message.content})

#         except Exception as e:
#             return Response({"error": str(e)}, status=500)




# worker===================================================================================
# import re
# import logging

# from rest_framework.views import APIView
# from rest_framework.response import Response
# from rest_framework.permissions import AllowAny
# from django.views.decorators.csrf import csrf_exempt
# from django.utils.decorators import method_decorator
# from openai import OpenAI

# logger = logging.getLogger(__name__)

# def clean_special_chars(text):
#     # Remove markdown bold/italic/code formatting
#     text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
#     text = re.sub(r'\*(.*?)\*', r'\1', text)
#     text = re.sub(r'`{1,3}(.*?)`{1,3}', r'\1', text)
#     text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)  # Remove headers (e.g., ##)
#     text = re.sub(r'[\u2600-\u26FF\u2700-\u27BF\uE000-\uF8FF]', '', text)  # Remove emojis
#     return text.strip()

# @method_decorator(csrf_exempt, name='dispatch')
# class DeepSeekChatView(APIView):
#     permission_classes = [AllowAny]

#     def post(self, request):
#         try:
#             data = request.data

#             symbol = data.get("symbol", "N/A")
#             name = data.get("name", "N/A")
#             query_type = data.get("queryType", "N/A")
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

#             prompt = f"""
# Act as an expert financial analyst and return your analysis in clear markdown format.

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
# Query Type: {query_type}  

# News Headlines  
# {news_lines}

# ## Key Financial Metrics  
# List valuation ratios, margins, ROE, and any known financial KPIs.

# ## Strategic Initiatives  
# Mention growth areas, innovations, or major company projects.

# ## Upcoming Events  
# Include earnings dates, estimates, and any financial releases.

# ## Analyst Insights  
# Summarize bullish/bearish factors, estimates, momentum, and sentiment.

# ## Risks  
# Highlight major financial, regulatory, or competitive risks.
# """

#             client = OpenAI(
#                 api_key="sk-fd092005f2f446d78dade7662a13c896",
#                 base_url="https://api.deepseek.com"
#             )

#             chat_response = client.chat.completions.create(
#                 model="deepseek-chat",
#                 messages=[
#                     {"role": "system", "content": "You are TradeGPT, a professional market analyst."},
#                     {"role": "user", "content": prompt}
#                 ],
#                 stream=False
#             )

#             raw = chat_response.choices[0].message.content

#             if not raw.lstrip().lower().startswith("company overview"):
#                 raw = "Company Overview\n\n" + raw

#             cleaned = clean_special_chars(raw)

#             return Response({"message": cleaned})

#         except Exception as e:
#             logger.error(f"DeepSeek error: {str(e)}")
#             return Response({"error": str(e)}, status=500)



import re
import logging

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import AllowAny
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from openai import OpenAI

logger = logging.getLogger(__name__)

def convert_markdown_to_html_sections(markdown):
    html_output = ""
    lines = markdown.splitlines()
    current_section = ""

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith("##"):
            if current_section:
                html_output += f"</div>"
            heading = re.sub(r'^#+\s*', '', line)
            html_output += f'<div style="margin-top:24px"><h3 style="font-weight:600">{heading}</h3>'
            current_section = heading
        elif re.match(r'^-\s', line):
            content = line[2:].strip()
            html_output += f'<p style="margin-left: 1rem">{content}</p>'
        elif ":" in line:
            parts = line.split(":", 1)
            key = parts[0].strip()
            value = parts[1].strip()
            html_output += f'<p><strong>{key}:</strong> {value}</p>'
        else:
            html_output += f"<p>{line}</p>"

    if current_section:
        html_output += "</div>"

    return html_output.strip()

@method_decorator(csrf_exempt, name='dispatch')
class DeepSeekChatView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        try:
            data = request.data

            symbol = data.get("symbol", "N/A")
            name = data.get("name", "N/A")
            query_type = data.get("queryType", "N/A")
            price = data.get("price", "N/A")
            open_ = data.get("open", "N/A")
            high = data.get("high", "N/A")
            low = data.get("low", "N/A")
            previous_close = data.get("previousClose", "N/A")
            volume = data.get("volume", "N/A")
            trend = data.get("trend", "N/A")
            news_list = data.get("news", [])

            news_lines = ""
            for item in news_list[:5]:
                headline = item.get("headline", "No headline")
                time_str = item.get("time", "Unknown time")
                category = item.get("category", "General")
                news_lines += f"- {headline} at {time_str} | {category}\n"

            if not news_lines.strip():
                news_lines = "No major headlines available."

            prompt = f"""
Act as an expert financial analyst and return your analysis in clear markdown format.

## Technical & Fundamental Analysis: {name} ({symbol})
**Price:** ${price} | **Trend:** {trend} | **Date:** 2025-05-27

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
Query Type: {query_type}  

News Headlines  
{news_lines}

## Key Financial Metrics  
List valuation ratios, margins, ROE, and any known financial KPIs.

## Strategic Initiatives  
Mention growth areas, innovations, or major company projects.

## Upcoming Events  
Include earnings dates, estimates, and any financial releases.

## Analyst Insights  
Summarize bullish/bearish factors, estimates, momentum, and sentiment.

## Risks  
Highlight major financial, regulatory, or competitive risks.
"""

            client = OpenAI(
                api_key="sk-fd092005f2f446d78dade7662a13c896",
                base_url="https://api.deepseek.com"
            )

            chat_response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are TradeGPT, a professional market analyst."},
                    {"role": "user", "content": prompt}
                ],
                stream=False
            )

            raw = chat_response.choices[0].message.content
            html_cleaned = convert_markdown_to_html_sections(raw)

            return Response({"message": html_cleaned})

        except Exception as e:
            logger.error(f"DeepSeek error: {str(e)}")
            return Response({"error": str(e)}, status=500)
