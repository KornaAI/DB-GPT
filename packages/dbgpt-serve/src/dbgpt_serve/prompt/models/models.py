"""This is an auto-generated model file
You can define your own models and DAOs here
"""

import uuid
from datetime import datetime
from typing import Any, Dict, Union

from sqlalchemy import Column, DateTime, Integer, String, Text, UniqueConstraint

from dbgpt.storage.metadata import BaseDao, Model

from ..api.schemas import ServeRequest, ServerResponse
from ..config import ServeConfig


class ServeEntity(Model):
    __tablename__ = "prompt_manage"
    __table_args__ = (
        UniqueConstraint(
            "prompt_name",
            "sys_code",
            "prompt_language",
            "model",
            name="uk_prompt_name_sys_code",
        ),
    )
    id = Column(Integer, primary_key=True, comment="Auto increment id")

    chat_scene = Column(String(100), comment="Chat scene")
    sub_chat_scene = Column(String(100), comment="Sub chat scene")
    prompt_code = Column(String(256), comment="Prompt Code")
    prompt_type = Column(String(100), comment="Prompt type(eg: common, private)")
    prompt_name = Column(String(256), comment="Prompt name")
    content = Column(Text, comment="Prompt content")
    input_variables = Column(
        String(1024), nullable=True, comment="Prompt input variables(split by comma))"
    )
    response_schema = Column(Text, comment="Prompt response schema")
    model = Column(
        String(128),
        nullable=True,
        comment="Prompt model name(we can use different models for different prompt",
    )
    prompt_language = Column(
        String(32), index=True, nullable=True, comment="Prompt language(eg:en, zh-cn)"
    )
    prompt_format = Column(
        String(32),
        index=True,
        nullable=True,
        default="f-string",
        comment="Prompt format(eg: f-string, jinja2)",
    )
    prompt_desc = Column(String(512), nullable=True, comment="Prompt description")
    user_code = Column(String(128), index=True, nullable=True, comment="User code")
    user_name = Column(String(128), index=True, nullable=True, comment="User name")
    sys_code = Column(String(128), index=True, nullable=True, comment="System code")
    gmt_created = Column(DateTime, default=datetime.now, comment="Record creation time")
    gmt_modified = Column(DateTime, default=datetime.now, comment="Record update time")

    def __repr__(self):
        return (
            f"ServeEntity(id={self.id}, chat_scene='{self.chat_scene}', "
            f"sub_chat_scene='{self.sub_chat_scene}', "
            f"prompt_type='{self.prompt_type}', prompt_name='{self.prompt_name}', "
            f"content='{self.content}',"
            f"user_code='{self.user_code}', user_name='{self.user_name}', "
            f"gmt_created='{self.gmt_created}', gmt_modified='{self.gmt_modified}')"
        )


class ServeDao(BaseDao[ServeEntity, ServeRequest, ServerResponse]):
    """The DAO class for Prompt"""

    def __init__(self, serve_config: ServeConfig):
        super().__init__()
        self._serve_config = serve_config

    def from_request(self, request: Union[ServeRequest, Dict[str, Any]]) -> ServeEntity:
        """Convert the request to an entity

        Args:
            request (Union[ServeRequest, Dict[str, Any]]): The request

        Returns:
            T: The entity
        """
        request_dict = request.dict() if isinstance(request, ServeRequest) else request
        entity = ServeEntity(**request_dict)
        if not entity.prompt_code:
            entity.prompt_code = uuid.uuid4().hex
        return entity

    def to_request(self, entity: ServeEntity) -> ServeRequest:
        """Convert the entity to a request

        Args:
            entity (T): The entity

        Returns:
            REQ: The request
        """
        return ServeRequest(
            chat_scene=entity.chat_scene,
            sub_chat_scene=entity.sub_chat_scene,
            prompt_type=entity.prompt_type,
            prompt_name=entity.prompt_name,
            prompt_code=entity.prompt_code,
            content=entity.content,
            response_schema=entity.response_schema,
            input_variables=entity.input_variables,
            model=entity.model,
            prompt_language=entity.prompt_language,
            prompt_desc=entity.prompt_desc,
            user_code=entity.user_code,
            user_name=entity.user_name,
            sys_code=entity.sys_code,
        )

    def to_response(self, entity: ServeEntity) -> ServerResponse:
        """Convert the entity to a response

        Args:
            entity (T): The entity

        Returns:
            RES: The response
        """
        # TODO implement your own logic here, transfer the entity to a response
        gmt_created_str = entity.gmt_created.strftime("%Y-%m-%d %H:%M:%S")
        gmt_modified_str = entity.gmt_modified.strftime("%Y-%m-%d %H:%M:%S")
        return ServerResponse(
            id=entity.id,
            chat_scene=entity.chat_scene,
            sub_chat_scene=entity.sub_chat_scene,
            prompt_code=entity.prompt_code,
            prompt_type=entity.prompt_type,
            prompt_name=entity.prompt_name,
            content=entity.content,
            prompt_desc=entity.prompt_desc,
            user_name=entity.user_name,
            user_code=entity.user_code,
            model=entity.model,
            input_variables=entity.input_variables,
            response_schema=entity.response_schema,
            prompt_language=entity.prompt_language,
            sys_code=entity.sys_code,
            gmt_created=gmt_created_str,
            gmt_modified=gmt_modified_str,
        )
