"""Read-only textbook catalogue, independent of retrieval and indexing."""
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session


def catalogue(engine=None):
    from app.db.tables import DocumentBlock
    if engine is None:
        from app.db.postgres import get_engine
        engine = get_engine()
    with Session(engine) as session:
        if engine.dialect.name == 'postgresql':
            session.execute(text('SET TRANSACTION READ ONLY'))
        rows = session.execute(select(
            DocumentBlock.doc_id, func.min(DocumentBlock.doc_type).label('doc_type'),
            func.count(func.distinct(DocumentBlock.doc_type)).label('type_count'),
            func.count(func.distinct(DocumentBlock.page)).label('pages'),
            func.count().label('blocks'),
        ).group_by(DocumentBlock.doc_id).order_by(DocumentBlock.doc_id)).mappings()
        result = []
        for row in rows:
            value = dict(row)
            if value.pop('type_count') > 1:
                value['doc_type'] = '混合类型（需核对）'
            result.append(value)
        return result


def material_preview(doc_id, *, engine=None):
    from app.db.tables import DocumentBlock, DocumentImportRun
    if engine is None:
        from app.db.postgres import get_engine
        engine = get_engine()
    with Session(engine) as session:
        if engine.dialect.name == 'postgresql':
            session.execute(text('SET TRANSACTION READ ONLY'))
        first_page = select(func.min(DocumentBlock.page)).where(DocumentBlock.doc_id == doc_id).scalar_subquery()
        blocks = session.execute(select(DocumentBlock.id,DocumentBlock.page,DocumentBlock.block_index,
            DocumentBlock.type,DocumentBlock.text,DocumentBlock.indexable,DocumentBlock.is_generated,DocumentBlock.source_kind)
            .where(DocumentBlock.doc_id == doc_id,DocumentBlock.page == first_page)
            .order_by(DocumentBlock.block_index).limit(12)).mappings()
        output = [dict(r) for r in blocks]
        runs = session.execute(select(DocumentImportRun.id,DocumentImportRun.status,DocumentImportRun.pages_spec,
            DocumentImportRun.pages_total,DocumentImportRun.pages_ok,DocumentImportRun.started_at)
            .where(DocumentImportRun.doc_id == doc_id).order_by(DocumentImportRun.started_at.desc()).limit(5)).mappings()
        return {'blocks':output,'runs':[dict(r) for r in runs]}
