from django.db import models

class ChatSession(models.Model):
    session_id = models.UUIDField(primary_key=True, editable=False, auto_created=True)
    user_id = models.IntegerField()
    username = models.CharField(max_length=150)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Session {self.session_id} - {self.username}"


class ChatMessage(models.Model):
    session = models.ForeignKey(ChatSession, on_delete=models.CASCADE, related_name="messages")
    role = models.CharField(max_length=10, choices=[("user", "User"), ("ai", "AI")])
    content = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)
