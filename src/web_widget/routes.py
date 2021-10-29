import re
from datetime import datetime, timedelta
from typing import List, Optional

from fastapi import APIRouter, Body, Depends, Query
from pydantic import EmailStr
from pytz import timezone
from starlette.responses import Response
from starlette.status import HTTP_200_OK, HTTP_404_NOT_FOUND

from .models import EnrollmentResponse, Event, Client, UpdateEnrollment
from .utils import initialize_mindbody

# Internal SDKs
from datacose.mindbody import MindBody

router = APIRouter(prefix='/web-widget')


@router.get(
    "/get-client",
    status_code=HTTP_200_OK,
    tags=["Web Widget"],
    response_model=Client
)
def get_client(*, user_mail: EmailStr = Query(..., alias="userMail"),
               user_phone: str = Query(..., alias="userPhone", min_length=9),
               mindbody: MindBody = Depends(initialize_mindbody)):
    clients: List[dict] = mindbody.client_api.get_clients(search_text=user_mail).json().get('Clients')

    def match_client(clients: List[dict]) -> Optional[dict]:
        formatted_user_phone = re.sub(r'\D', '', user_phone)

        def _match(client: dict) -> Optional[bool]:
            if client.get('Email') != user_mail:
                return False
            for field, value in client.items():
                if not value:
                    continue
                if 'phone' in field.lower() and formatted_user_phone in re.sub(r'\D', '', value):
                    return True
            else:
                return False

        for client in clients:
            if _match(client):
                return client
        return None

    if not (matched_client := match_client(clients=clients)):
        return Response(status_code=HTTP_404_NOT_FOUND)

    return Client(id=matched_client.get('Id'),
                  first_name=matched_client.get('FirstName'),
                  last_name=matched_client.get('LastName')
                  )


@router.get(
    "/get-upcoming-events",
    status_code=HTTP_200_OK,
    tags=["Web Widget"],
    response_model=List[Event]
)
def get_upcoming_events(*, current_client_id: str = Query(..., alias="currentClientId"),
                        company_client_id: str = Query(..., alias="companyClientId"),
                        start_date: datetime = Query(datetime.now(tz=timezone('US/Eastern')) + timedelta(minutes=10),
                                                     alias='startDate'),
                        end_date: datetime = Query(datetime.now(tz=timezone('US/Eastern')) + timedelta(days=60),
                                                   alias='endDate'),
                        mindbody: MindBody = Depends(initialize_mindbody)):
    company_visits: List[dict] = mindbody.client_api.get_client_visits(
        client_id=company_client_id,
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
    ).json().get('Visits')

    if not company_visits:
        return []

    company_classes_id: List[int] = [visit.get('ClassId') for visit in company_visits]

    company_classes: List[dict] = mindbody.class_api.get_classes(
        class_ids=company_classes_id,
        start_date_time=start_date.isoformat(),
        end_date_time=end_date.isoformat()
    ).json().get('Classes')

    current_client_visits: List[dict] = mindbody.client_api.get_client_visits(
        client_id=current_client_id,
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat()
    ).json().get('Visits')

    current_client_classes_id: List[int] = [visit.get('ClassId') for visit in current_client_visits]

    upcoming_events: List[Event] = []
    for company_class in company_classes:
        event = {
            'id': company_class.get('Id'),
            'name': company_class.get('ClassDescription').get('Name'),
            'instructor': f"{company_class.get('Staff').get('FirstName')} {company_class.get('Staff').get('LastName')}",
            'start_time': company_class.get('StartDateTime'),
            'end_time': company_class.get('EndDateTime'),
            'description': company_class.get('ClassDescription').get('Description'),
            'current_user_enrolled': company_class.get('Id') in current_client_classes_id,
        }
        upcoming_events.append(Event(**event))

    return upcoming_events


@router.post(
    "/update-enrollment",
    status_code=HTTP_200_OK,
    tags=["Web Widget"],
    response_model=EnrollmentResponse
)
def update_enrolment(update_enrollment: UpdateEnrollment = Body(...),
                     mindbody: MindBody = Depends(initialize_mindbody)):
    if update_enrollment.enroll:
        update = mindbody.class_api.add_client
    else:
        update = mindbody.class_api.remove_client

    update(client_id=update_enrollment.client_id, class_id=update_enrollment.class_id)

    return EnrollmentResponse(client_id=update_enrollment.client_id,
                              class_id=update_enrollment.class_id,
                              enrolled=update_enrollment.enroll)
