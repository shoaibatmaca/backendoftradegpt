import jwt
from django.conf import settings
from rest_framework.exceptions import AuthenticationFailed

def get_user_from_token(token):
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=["HS256"])
        return {
            "user_id": payload.get("user_id"),
            "username": payload.get("username"),
            "subscription_status": payload.get("subscription_status", "free")
        }
    except jwt.ExpiredSignatureError:
        raise AuthenticationFailed("Token expired")
    except jwt.DecodeError:
        raise AuthenticationFailed("Invalid token")
