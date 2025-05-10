# from django.shortcuts import render
# from rest_framework.parsers import MultiPartParser, FormParser
# from rest_framework.views import APIView
# from rest_framework.response import Response
# from rest_framework.permissions import IsAuthenticated, AllowAny
# import jwt
# from datetime import datetime, timedelta
# from django.conf import settings

# from rest_framework import status

# class TradeGPTUserView(APIView):
#     permission_classes = [AllowAny]

#     def get(self, request):
#         token = request.GET.get("token")
#         if not token:
#             return Response({"error": "Token is missing"}, status=400)

#         try:
#             decoded = jwt.decode(token, settings.SECRET_KEY, algorithms=["HS256"])
#             return Response({
#                 "user_id": decoded.get("user_id"),
#                 "username": decoded.get("username"),
#                 "first_name": decoded.get("first_name"),
#                 "last_name": decoded.get("last_name"),
#                 "email": decoded.get("email"),
#                 "subscription_status": decoded.get("subscription_status"),
#                 "profile_photo": decoded.get("profile_photo"),
#                 "phone_number": decoded.get("phone_number"),
#                 "country": decoded.get("country"),
#                 "state": decoded.get("state"),
#                 "is_staff": decoded.get("is_staff"),
#                 "is_superuser": decoded.get("is_superuser"),
#             })
#         except jwt.ExpiredSignatureError:
#             return Response({"error": "Token expired"}, status=401)
#         except jwt.InvalidTokenError:
#             return Response({"error": "Invalid token"}, status=401)





# from rest_framework.views import APIView
# from rest_framework.response import Response
# from rest_framework import status
# from .models import ChatSession, ChatMessage
# from .utils import get_user_from_token
# from datetime import datetime
# from django.utils.timezone import now
# import uuid
# class StartChatSessionView(APIView):
#     def get(self, request):
#         return Response({"detail": "GET not allowed."}, status=405)

#     def post(self, request):
#         token = request.GET.get("token")
#         user = get_user_from_token(token)

#         session = ChatSession.objects.create(
#             session_id=uuid.uuid4(),
#             user_id=user["user_id"],
#             username=user["username"],
#         )
#         return Response({"session_id": session.session_id})



# class MessageListCreateView(APIView):
#     def post(self, request, session_id):
#         token = request.GET.get("token")
#         user = get_user_from_token(token)

#         data = request.data
#         ChatMessage.objects.create(
#             session_id=session_id,
#             role=data["role"],
#             content=data["content"]
#         )
#         return Response({"message": "Saved"}, status=201)

#     def get(self, request, session_id):
#         token = request.GET.get("token")
#         user = get_user_from_token(token)

#         messages = ChatMessage.objects.filter(session_id=session_id).order_by("timestamp")
#         return Response([
#             {"role": m.role, "content": m.content, "timestamp": m.timestamp}
#             for m in messages
#         ])


# class UserChatSessionsView(APIView):
#     def get(self, request):
#         token = request.GET.get("token")
#         user = get_user_from_token(token)

#         sessions = ChatSession.objects.filter(user_id=user["user_id"]).order_by("-created_at")
#         return Response([
#             {"session_id": s.session_id, "created_at": s.created_at}
#             for s in sessions
#         ])


# class DailyMessageLimitView(APIView):
#     def get(self, request):
#         token = request.GET.get("token")
#         user = get_user_from_token(token)

#         count = ChatMessage.objects.filter(
#             session__user_id=user["user_id"],
#             timestamp__date=now().date()
#         ).count()

#         max_allowed = {
#             "free": 3,
#             "premium": 5,
#             "platinum": 10,
#         }.get(user["subscription_status"], 3)

#         return Response({"count": count, "max": max_allowed})

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




import requests
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import AllowAny
from django.conf import settings

# class OpenRouterProxyView(APIView):
#     permission_classes = [AllowAny]

#     def post(self, request):
#         prompt = request.data.get("prompt")
#         if not prompt:
#             return Response({"error": "Missing prompt"}, status=400)

#         headers = {
#             "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",  # safer in env
#             "Content-Type": "application/json",
#             "HTTP-Referer": "https://frontend-eight-rho-95.vercel.app",  # your frontend URL
#             "X-Title": "TradeGPT Chat"
#         }

#         payload = {
#             "model": "deepseek/deepseek-chat:free",
#             "messages": [{"role": "user", "content": prompt}]
#         }

#         try:
#             res = requests.post("https://openrouter.ai/api/v1/chat/completions", json=payload, headers=headers)
#             return Response(res.json(), status=res.status_code)
#         except Exception as e:
#             return Response({"error": str(e)}, status=500)

class OpenRouterProxyView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        prompt = request.data.get("prompt")
        if not prompt:
            return Response({"error": "Missing prompt"}, status=400)

        api_key = settings.OPENROUTER_API_KEY
        if not api_key:
            return Response({"error": "API key missing from env"}, status=500)

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://frontend-eight-rho-95.vercel.app",
            "X-Title": "TradeGPT Chat"
        }

        payload = {
            "model": "deepseek/deepseek-chat:free",
            "messages": [{"role": "user", "content": prompt}]
        }

        try:
            res = requests.post("https://openrouter.ai/api/v1/chat/completions", json=payload, headers=headers)
            return Response(res.json(), status=res.status_code)
        except Exception as e:
            return Response({"error": str(e)}, status=500)
