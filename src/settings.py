import os
from pathlib import Path
import attr
import typing as T
from attr.validators import instance_of, optional
from dotenv import load_dotenv

load_dotenv()
@attr.s
class Config:
    CONSUMER_KEY: T.Optional[str] = attr.ib(validator=optional(instance_of(str)),
                                            default=os.getenv("TWITTER_API_KEY"))
    CONSUMER_SECRET: T.Optional[str] = attr.ib(validator=optional(instance_of(str)),
                                               default=os.getenv("TWITTER_API_SECRET"))
    ACCESS_KEY: T.Optional[str] = attr.ib(validator=optional(instance_of(str)),
                                          default=os.getenv("TWITTER_API_ACCESS_TOKEN"))
    ACCESS_TOKEN_SECRET: T.Optional[str] = attr.ib(validator=optional(instance_of(str)),
                                                   default=os.getenv("TWITTER_API_ACCESS_TOKEN_SECRET"))
    DATA_DIR: T.Optional[Path] = attr.ib(validator=optional(instance_of(Path)),
                                         default=Path(__file__).parent.parent.absolute() / 'data')