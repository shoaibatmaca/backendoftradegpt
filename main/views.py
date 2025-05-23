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




# import requests
# from rest_framework.views import APIView
# from rest_framework.response import Response
# from rest_framework.permissions import AllowAny
# from django.conf import settings


# import requests
# from rest_framework.views import APIView
# from rest_framework.response import Response
# from rest_framework.permissions import AllowAny
# from django.conf import settings


# import requests
# import re
# from rest_framework.views import APIView
# from rest_framework.response import Response
# from rest_framework.permissions import AllowAny
# from .utils import get_user_from_token 

# from django.views.decorators.csrf import csrf_exempt
# from django.utils.decorators import method_decorator

# @method_decorator(csrf_exempt, name='dispatch')
# class OpenRouterProxyView(APIView):
#     permission_classes = [AllowAny]

#     def post(self, request):
#         from .utils import get_user_from_token
#         import requests, re

#         token = request.GET.get("token")
#         if not token:
#             return Response({"error": "Token is missing"}, status=400)

#         try:
#             user = get_user_from_token(token)
#         except Exception as e:
#             return Response({"error": str(e)}, status=401)

#         model = request.data.get("model")
#         messages = request.data.get("messages")
#         stream = request.data.get("stream", False)
#         metadata = request.data.get("metadata", {})

#         if not model or not messages:
#             return Response({"error": "Missing model or messages"}, status=400)

#         payload = {
#             "model": model,
#             "messages": messages,
#             "stream": stream,
#             "options": {
#                 "temperature": 0.7,
#                 "top_k": 40
#             }
#         }

#         try:
#             res = requests.post(
#             "https://6e07-39-49-168-243.ngrok-free.app/api/chat",
#                 json=payload,
#                 headers={"Content-Type": "application/json"},
#                 timeout=60
#             )

#             res_data = res.json()
#             for key in ["message", "content", "response"]:
#                 if key in res_data and isinstance(res_data[key], str):
#                     res_data[key] = re.sub(r"<think>.*?</think>", "", res_data[key], flags=re.DOTALL)

#             return Response(res_data, status=res.status_code)

#         except Exception as e:
#             return Response({"error": f"Ollama Local Error: {str(e)}"}, status=500)

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import AllowAny
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from .utils import get_user_from_token
import requests
import re

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
        messages = request.data.get("messages")
        stream = request.data.get("stream", False)

        if not model or not messages:
            return Response({"error": "Missing model or messages"}, status=400)

        # Convert OpenAI-style messages into a single prompt for Ollama
        prompt = "\n\n".join([f"{m['role'].upper()}: {m['content']}" for m in messages])

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
        }

        try:
            res = requests.post(
                "https://d955-119-156-137-152.ngrok-free.app/api/generate",  # Correct Ollama endpoint
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=60
            )
            res.raise_for_status()
            res_data = res.json()

            if "response" in res_data and isinstance(res_data["response"], str):
                res_data["response"] = re.sub(r"<think>.*?</think>", "", res_data["response"], flags=re.DOTALL)

            return Response({"message": res_data.get("response", "")}, status=res.status_code)

        except requests.exceptions.RequestException as e:
            return Response({"error": f"Ollama request failed: {str(e)}"}, status=500)

        except Exception as e:
            return Response({"error": f"Unexpected error: {str(e)}"}, status=500)
