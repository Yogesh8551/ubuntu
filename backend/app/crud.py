from sqlalchemy.orm import Session
from sqlalchemy import func
from .models import Resume

def create_resume(db: Session, **data):
    obj = Resume(**data)
    db.add(obj)
    db.commit()
    db.refresh(obj)
    return obj

def query_resumes(db: Session, name=None, resumetype=None, occupations=None):
    # """
    # occupations: comma-separated string like "devops,civil"
    # """
    q = db.query(Resume)

    # Exact match for multiple occupations (case-insensitive)
    if occupations:
        occupation_list = [occ.strip().lower() for occ in occupations.split(",")]
        q = q.filter(func.lower(func.trim(Resume.occupation)).in_(occupation_list))

    # Optional: exact match for name/resumetype (partial match)
    if name:
        q = q.filter(Resume.name.ilike(f"%{name}%"))
    if resumetype:
        q = q.filter(Resume.resumetype.ilike(f"%{resumetype}%"))

    return q.all()
