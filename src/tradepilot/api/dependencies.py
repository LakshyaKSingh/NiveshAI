"""
FastAPI Dependencies: DB Sessions and RBAC Authentication
=========================================================
"""

from typing import Optional, List, Union
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from tradepilot.db.session import get_db
from tradepilot.db.models import User
from tradepilot.api.security import decode_access_token
from tradepilot.api.config import settings

oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl=f"{settings.API_V1_PREFIX}/auth/login",
    auto_error=False,
)


def get_current_user(
    token: Optional[str] = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> User:
    """Validate Bearer token and return active User instance."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials or token expired",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if not token:
        raise credentials_exception

    payload = decode_access_token(token)
    if not payload:
        raise credentials_exception

    user_id: Optional[int] = payload.get("sub") or payload.get("user_id")
    if user_id is None:
        raise credentials_exception

    user = db.query(User).filter(User.id == int(user_id)).first()
    if not user:
        raise credentials_exception
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user account",
        )
    return user


def get_current_active_user(
    current_user: User = Depends(get_current_user),
) -> User:
    """Convenience dependency returning an authenticated active user."""
    return current_user


def require_role(allowed_roles: Union[str, List[str]]):
    """Enforce role-based access control (RBAC)."""
    if isinstance(allowed_roles, str):
        allowed_roles = [allowed_roles]

    def role_checker(current_user: User = Depends(get_current_user)) -> User:
        # Admin can access all role-gated endpoints
        if current_user.role == "admin" or current_user.role in allowed_roles:
            return current_user
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Operation requires one of the following roles: {allowed_roles}. Current role: {current_user.role}",
        )

    return role_checker
