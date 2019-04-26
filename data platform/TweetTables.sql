



--Table: public.TwitterUser

--DROP TABLE public."TwitterUser";

CREATE TABLE public."TwitterUser" (
  id                bigint NOT NULL PRIMARY KEY,
  "name"            text,
  followers_count   bigint,
  friends_count     bigint,
  listed_count      bigint,
  favourites_count  bigint,
  statuses_count    bigint,
  created_at        date,
  lang              varchar(30),
  /* Keys */
  CONSTRAINT "TwitterUser_pkey"
    PRIMARY KEY (id)
) WITH (
    OIDS = FALSE
  );

ALTER TABLE public."TwitterUser"
  OWNER TO postgres;



--Table: public.Tweet

--DROP TABLE public."Tweet";

CREATE TABLE public."Tweet" (
  id              bigint NOT NULL,
  "text"          text,
  quote_count     bigint,
  reply_count     bigint,
  retweet_count   bigint,
  favorite_count  bigint,
  lang            varchar(30),
  "userId"        bigint,
  /* Keys */
  CONSTRAINT "Tweets_pkey"
    PRIMARY KEY (id),
  /* Foreign keys */
  CONSTRAINT "userIdFk"
    FOREIGN KEY ("userId")
    REFERENCES public."TwitterUser"(id)
) WITH (
    OIDS = FALSE
  );

ALTER TABLE public."Tweet"
  OWNER TO postgres;

