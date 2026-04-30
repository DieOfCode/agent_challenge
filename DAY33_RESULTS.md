# Day 33 Results: Support Assistant

- transport: `inprocess`
- server: `day33-inprocess-support-mcp`
- server version: `1.0.0`
- protocol: `2025-11-25`
- ticket: `#5001`
- user: `#101`
- mode: `simulate`

## Question
Почему не работает авторизация?

## Answer
Краткий ответ:
Проблема похожа на сбой в auth-потоке (токен/сессия/2FA/SSO).

Почему:
В тикете есть auth-признаки: last_error=invalid_grant, code=AUTH_401.

Шаги для пользователя:
- Попросить пользователя выйти из всех сессий и войти повторно.
- Проверить корректность времени на устройстве и актуальность метода входа.

Шаги для поддержки:
- Проверить auth-логи по request_id/времени попытки.
- Проверить лимиты попыток, статус 2FA и состояние SSO-конфига.
- Учесть профиль пользователя: plan=pro, auth_provider=email_password.

Источники:
- docs/SUPPORT_FAQ.md (docs/SUPPORT_FAQ.md#000, 0.635); docs/SUPPORT_FAQ.md (docs/SUPPORT_FAQ.md#001, 0.240); docs/SUPPORT_FAQ.md (docs/SUPPORT_FAQ.md#003, 0.081); docs/SUPPORT_FAQ.md (docs/SUPPORT_FAQ.md#002, 0.081)

[simulate] question=Почему не работает авторизация?

## Sources
- `docs/SUPPORT_FAQ.md` (docs/SUPPORT_FAQ.md#000, score=0.6351)
- `docs/SUPPORT_FAQ.md` (docs/SUPPORT_FAQ.md#001, score=0.2405)
- `docs/SUPPORT_FAQ.md` (docs/SUPPORT_FAQ.md#003, score=0.0806)
- `docs/SUPPORT_FAQ.md` (docs/SUPPORT_FAQ.md#002, score=0.0805)
