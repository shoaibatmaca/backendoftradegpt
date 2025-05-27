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




from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import AllowAny
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
import requests

@method_decorator(csrf_exempt, name='dispatch')
class DeepSeekChatView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        # receive: symbol, name, open, high, low, volume, previousClose, news[], queryType
        data = request.data
        symbol = data.get("symbol")
        name = data.get("name")
        query_type = data.get("queryType")

        prompt = f"""
        Act as an expert financial analyst. Provide a structured analysis for the following ticker based on the context.

        ▶ Symbol: {symbol}
        ▶ Company: {name}
        ▶ Price: ${data.get("price")}
        ▶ Open: ${data.get("open")}
        ▶ High: ${data.get("high")}
        ▶ Low: ${data.get("low")}
        ▶ Previous Close: ${data.get("previousClose")}
        ▶ Volume: {data.get("volume")}
        ▶ Trend: {data.get("trend")}
        ▶ Query Type: {query_type}

        📰 Top News:
        {"".join([f"- {n['headline']} at {n['time']} | {n['category']}\n" for n in data.get("news", [])])}

        Based on the above, give a response for: "{query_type}". Include company overview, key ratios, strategic initiatives, upcoming events, risks, and trade suggestions if applicable.
        """

        headers = {
            "Authorization": "Bearer sk-fd092005f2f446d78dade7662a13c896",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": "You are TradeGPT, a professional market analyst."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "stream": False
        }

        try:
            res = requests.post("https://api.deepseek.com/v1/chat/completions", headers=headers, json=payload)
            res.raise_for_status()
            result = res.json()
            return Response({"message": result["choices"][0]["message"]["content"]})
        except Exception as e:
            return Response({"error": str(e)}, status=500)
