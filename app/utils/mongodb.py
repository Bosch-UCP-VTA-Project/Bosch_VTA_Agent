from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient
from typing import Optional, List, Dict
import os
from datetime import datetime
from bson import ObjectId


class MongoDB:
    def __init__(self, uri=None):
        self.uri = uri or os.getenv("MONGODB_URI")
        if not self.uri:
            raise ValueError("MONGODB_URI environment variable is not set")

        # Create sync and async clients
        self.client = MongoClient(self.uri)
        self.async_client = AsyncIOMotorClient(self.uri)

        # Define database
        self.db = self.client.bosch_vta
        self.async_db = self.async_client.bosch_vta

        # Define collections
        self.users = self.db.users
        self.chats = self.db.chats
        self.sessions = self.db.sessions
        self.documents = self.db.documents

        # Async collections
        self.async_users = self.async_db.users
        self.async_chats = self.async_db.chats
        self.async_sessions = self.async_db.sessions
        self.async_documents = self.async_db.documents

    # User operations
    async def get_user_by_username(self, username: str) -> Optional[Dict]:
        return await self.async_users.find_one({"username": username})

    async def get_user_by_id(self, user_id: str) -> Optional[Dict]:
        return await self.async_users.find_one({"id": user_id})

    async def create_user(self, user_data: Dict) -> str:
        result = await self.async_users.insert_one(user_data)
        return str(result.inserted_id)

    # Chat history operations
    async def get_chat_history(self, session_id: str) -> List[Dict]:
        cursor = self.async_chats.find({"session_id": session_id}).sort("timestamp", 1)
        return await cursor.to_list(length=None)

    async def add_chat_message(self, message_data: Dict) -> str:
        result = await self.async_chats.insert_one(message_data)
        return str(result.inserted_id)

    async def get_first_message(self, session_id: str) -> Optional[Dict]:
        """Get the first message in a chat session"""
        return await self.async_chats.find_one(
            {"session_id": session_id}, sort=[("timestamp", 1)]
        )

    # Session operations
    async def get_session(self, session_id: str) -> Optional[Dict]:
        return await self.async_sessions.find_one({"session_id": session_id})

    async def create_session(self, session_data: Dict) -> str:
        result = await self.async_sessions.insert_one(session_data)
        return str(result.inserted_id)

    async def get_user_sessions(self, user_id: str) -> List[Dict]:
        cursor = self.async_sessions.find({"user_id": user_id}).sort("created_at", -1)
        return await cursor.to_list(length=None)

    # Document operations
    async def add_document(self, document_data: Dict) -> str:
        result = await self.async_documents.insert_one(document_data)
        return str(result.inserted_id)

    async def list_documents(self) -> List[Dict]:
        cursor = self.async_documents.find().sort("upload_date", -1)
        return await cursor.to_list(length=None)

    # Close connections
    def close(self):
        self.client.close()
        self.async_client.close()
