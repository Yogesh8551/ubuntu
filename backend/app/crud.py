from sqlalchemy.orm import Session
from sqlalchemy import func
from .models import Resume

import logging
logger = logging.getLogger("CRUD_LOGS")

def create_resume(db: Session, **data):
    obj = Resume(**data)
    db.add(obj)
    db.commit()
    db.refresh(obj)
    return obj

def query_resumes(db: Session, name=None, resumetype=None, occupation=None):

    q = db.query(Resume)

    # occupation filter - exact match ignoring case + trimming
    if occupation:
        q = q.filter(func.lower(func.trim(Resume.occupation)) == occupation.lower().strip())

    # name filter (partial)
    if name:
        q = q.filter(Resume.name.ilike(f"%{name}%"))

    # resume type filter (partial)
    if resumetype:
        q = q.filter(Resume.resumetype.ilike(f"%{resumetype}%"))

    return q.all()
