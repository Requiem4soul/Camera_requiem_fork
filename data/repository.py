from data.models import Rating
from data.db import get_session
from sqlalchemy import func
from sqlalchemy import desc


class RatingRepository:
    def add_rating(self, phone_model, metrics, analysis_method):
        session = get_session()
        try:
            # Проверяем существующие записи с таким phone_model
            existing_count = (
                session.query(Rating)
                .filter(
                    Rating.phone_model.ilike(phone_model),
                    Rating.analysis_method == analysis_method,
                )
                .count()
            )
            if existing_count > 0:
                # Если модель существует, добавляем суффикс (1), (2) и т.д.
                new_phone_model = f"{phone_model} ({existing_count})"
            else:
                # Если модели нет, используем оригинальное имя
                new_phone_model = phone_model

            total_score = sum(metrics.values()) / len(metrics) if metrics else None
            rating = Rating(
                phone_model=new_phone_model,
                analysis_method=analysis_method,
                sharpness=metrics.get("sharpness"),
                noise=metrics.get("noise"),
                glare=metrics.get("glare"),
                vignetting=metrics.get("vignetting"),
                chromatic_aberration=metrics.get("chromatic_aberration"),
                # Цветовые метрики
                color_gamut=metrics.get("color_gamut"),
                white_balance=metrics.get("white_balance"),
                contrast_ratio=metrics.get("contrast_ratio"),
                total_score=total_score,
            )
            session.add(rating)
            session.commit()
            return rating
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def get_average_ratings(self, analysis_method=None):
        session = get_session()
        try:
            query = session.query(
                Rating.phone_model,
                Rating.sharpness,
                Rating.noise,
                Rating.glare,
                Rating.chromatic_aberration,
                Rating.vignetting,
                # Цветовые метрики
                Rating.color_gamut,
                Rating.white_balance,
                Rating.contrast_ratio,
                Rating.total_score,
            )

            if analysis_method:
                query = query.filter(Rating.analysis_method == analysis_method)

            results = query.order_by(desc(Rating.total_score)).all()

            return [
                {
                    "phone_model": r.phone_model,
                    "sharpness": r.sharpness,
                    "noise": r.noise,
                    "glare": r.glare,
                    "chromatic_aberration": r.chromatic_aberration,
                    "vignetting": r.vignetting,
                    # Цветовые метрики
                    "color_gamut": r.color_gamut,
                    "white_balance": r.white_balance,
                    "contrast_ratio": r.contrast_ratio,
                    "total_score": r.total_score,
                }
                for r in results
            ]
        finally:
            session.close()
