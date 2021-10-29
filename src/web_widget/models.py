from fastapi_utils.api_model import APIModel


class Event(APIModel):
    id: int
    name: str
    instructor: str
    start_time: str
    end_time: str
    description: str
    current_user_enrolled: bool


class Client(APIModel):
    id: str
    first_name: str
    last_name: str


class Enrollment(APIModel):
    client_id: str
    class_id: int


class UpdateEnrollment(Enrollment):
    enroll: bool


class EnrollmentResponse(Enrollment):
    enrolled: bool

