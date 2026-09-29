"""Програмні правила перевірки вилучених із рахунку даних."""

import json
import re
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path

REFERENCE_DIR = Path(__file__).resolve().parent.parent / "reference"

# Для грошових розрахунків допускаємо різницю в 1 копійку.
MONEY_TOLERANCE = Decimal("0.01")


@dataclass
class Issue:
    """Проблема, знайдена програмним правилом."""

    field: str
    rule: str
    message: str
    severity: str = "error"

    @property
    def code(self) -> str:
        """Код перевірки у форматі samples/expected.json."""

        if self.rule in {"required", "money"}:
            return "missing_required"

        if self.rule == "item_arithmetic":
            return "arithmetic_line"

        if self.rule == "iban":
            return "iban_checksum"

        if (
            self.rule == "supplier_reference"
            and self.field == "supplier.iban"
        ):
            return "iban_registry"

        if self.rule == "buyer":
            return "buyer"

        if self.rule == "document_type":
            return "document_type"

        return self.rule


def load_reference() -> tuple[dict, list[dict]]:
    """Прочитати реквізити компанії й довідник постачальників."""

    company = json.loads(
        (REFERENCE_DIR / "company.json").read_text(encoding="utf-8")
    )

    suppliers = json.loads(
        (REFERENCE_DIR / "suppliers.json").read_text(encoding="utf-8")
    )

    return company, suppliers


def normalize_text(value) -> str:
    """Нормалізувати текст для надійного порівняння."""

    if value is None:
        return ""

    text = str(value).strip().lower()

    # Різні види апострофів -> один символ
    text = (
        text.replace("’", "'")
        .replace("ʼ", "'")
        .replace("`", "'")
        .replace("´", "'")
    )

    # Прибираємо різні лапки
    text = (
        text.replace("«", "")
        .replace("»", "")
        .replace('"', "")
        .replace("“", "")
        .replace("”", "")
    )

    # Нормалізуємо пробіли
    text = " ".join(text.split())

    return text


def normalize_code(value) -> str:
    """Залишити в коді тільки цифри."""

    if value is None:
        return ""

    return re.sub(r"\D", "", str(value))


def normalize_iban(value) -> str:
    """Прибрати пробіли та перевести IBAN у верхній регістр."""

    if value is None:
        return ""

    return re.sub(r"\s+", "", str(value)).upper()


def money(value) -> Decimal | None:
    """Безпечно перетворити грошове значення в Decimal."""

    if value is None:
        return None

    text = str(value).strip()

    if not text:
        return None

    # Прибираємо звичайні та нерозривні пробіли.
    text = text.replace(" ", "").replace("\u00a0", "")

    # Український десятковий роздільник.
    text = text.replace(",", ".")

    # Прибираємо поширені позначення валюти.
    text = re.sub(
        r"(?i)(грн\.?|uah|₴)",
        "",
        text,
    ).strip()

    try:
        return Decimal(text)
    except (InvalidOperation, ValueError):
        return None


def money_equal(a: Decimal, b: Decimal) -> bool:
    """Порівняти грошові суми з допуском 1 копійка."""

    return abs(a - b) <= MONEY_TOLERANCE


def round_money(value: Decimal) -> Decimal:
    """Округлити до копійок."""

    return value.quantize(
        Decimal("0.01"),
        rounding=ROUND_HALF_UP,
    )


# ---------------------------------------------------------
# IBAN
# ---------------------------------------------------------

def valid_iban(value: str | None) -> bool:
    """Перевірити український IBAN за ISO 13616."""

    iban = normalize_iban(value)

    # Український IBAN має 29 символів.
    if not re.fullmatch(r"UA\d{27}", iban):
        return False

    # Переносимо перші 4 символи в кінець.
    rearranged = iban[4:] + iban[:4]

    converted_parts = []

    for char in rearranged:
        if char.isdigit():
            converted_parts.append(char)
        elif "A" <= char <= "Z":
            converted_parts.append(str(ord(char) - ord("A") + 10))
        else:
            return False

    converted = "".join(converted_parts)

    # Рахуємо mod 97 поступово, не створюючи величезне число.
    remainder = 0

    for char in converted:
        remainder = (remainder * 10 + int(char)) % 97

    return remainder == 1


# ---------------------------------------------------------
# ЄДРПОУ
# ---------------------------------------------------------

def valid_edrpou(value: str | None) -> bool:
    """Перевірити контрольний розряд 8-значного ЄДРПОУ."""

    code = normalize_code(value)

    if len(code) != 8:
        return False

    digits = [int(x) for x in code]

    number = int(code)

    # Алгоритм має два набори ваг.
    if 30_000_000 <= number <= 60_000_000:
        weights1 = [7, 1, 2, 3, 4, 5, 6]
        weights2 = [9, 3, 4, 5, 6, 7, 8]
    else:
        weights1 = [1, 2, 3, 4, 5, 6, 7]
        weights2 = [3, 4, 5, 6, 7, 8, 9]

    checksum = sum(
        digit * weight
        for digit, weight in zip(digits[:7], weights1)
    ) % 11

    if checksum >= 10:
        checksum = sum(
            digit * weight
            for digit, weight in zip(digits[:7], weights2)
        ) % 11

    if checksum >= 10:
        checksum = 0

    return checksum == digits[7]


# ---------------------------------------------------------
# РНОКПП
# ---------------------------------------------------------

def valid_rnokpp(value: str | None) -> bool:
    """Перевірити 10-значний РНОКПП."""

    code = normalize_code(value)

    if len(code) != 10:
        return False

    digits = [int(x) for x in code]

    weights = [-1, 5, 7, 9, 4, 6, 10, 5, 7]

    checksum = sum(
        digit * weight
        for digit, weight in zip(digits[:9], weights)
    ) % 11

    checksum %= 10

    return checksum == digits[9]


def valid_company_code(value: str | None) -> bool:
    """Перевірити ЄДРПОУ або РНОКПП."""

    code = normalize_code(value)

    if len(code) == 8:
        return valid_edrpou(code)

    if len(code) == 10:
        return valid_rnokpp(code)

    return False


# ---------------------------------------------------------
# ДАТИ
# ---------------------------------------------------------

def parse_date(value) -> date | None:
    """Перетворити YYYY-MM-DD на date."""

    if value is None:
        return None

    try:
        return date.fromisoformat(str(value))
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------
# ОСНОВНА ПЕРЕВІРКА
# ---------------------------------------------------------

def check(document: dict) -> list[Issue]:
    """Застосувати всі програмні правила до документа."""

    issues: list[Issue] = []

    company, suppliers = load_reference()

    # -----------------------------------------------------
    # 1. Тип документа
    # -----------------------------------------------------

    if document.get("document_type") != "invoice":
        issues.append(
            Issue(
                field="document_type",
                rule="document_type",
                message="Документ не є рахунком на оплату.",
            )
        )

        # Інші перевірки для не-рахунку не мають сенсу.
        return issues

    # -----------------------------------------------------
    # 2. Обов'язкові поля
    # -----------------------------------------------------

    required_fields = [
        ("invoice_number", document.get("invoice_number")),
        ("invoice_date", document.get("invoice_date")),
        ("supplier.name", document.get("supplier", {}).get("name")),
        ("supplier.code", document.get("supplier", {}).get("code")),
        ("supplier.iban", document.get("supplier", {}).get("iban")),
        ("buyer.name", document.get("buyer", {}).get("name")),
        ("buyer.code", document.get("buyer", {}).get("code")),
        ("subtotal", document.get("subtotal")),
        ("vat", document.get("vat")),
        ("total", document.get("total")),
    ]

    for field, value in required_fields:
        if value is None or str(value).strip() == "":
            issues.append(
                Issue(
                    field=field,
                    rule="required",
                    message="Обов'язкове поле відсутнє або не читається.",
                )
            )

    items = document.get("items", [])

    if not items:
        issues.append(
            Issue(
                field="items",
                rule="required",
                message="У рахунку відсутні позиції.",
            )
        )

    # -----------------------------------------------------
    # 3. IBAN
    # -----------------------------------------------------

    supplier = document.get("supplier", {})
    supplier_iban = supplier.get("iban")

    if supplier_iban is not None and not valid_iban(supplier_iban):
        issues.append(
            Issue(
                field="supplier.iban",
                rule="iban",
                message=(
                    "IBAN має неправильний формат або "
                    "не проходить перевірку mod 97."
                ),
            )
        )

    # -----------------------------------------------------
    # 4. Коди постачальника та покупця
    # -----------------------------------------------------

    supplier_code = supplier.get("code")

    if (
        supplier_code is not None
        and not valid_company_code(supplier_code)
    ):
        issues.append(
            Issue(
                field="supplier.code",
                rule="company_code",
                message=(
                    "Код постачальника не проходить "
                    "перевірку контрольного розряду."
                ),
            )
        )

    buyer = document.get("buyer", {})
    buyer_code = buyer.get("code")

    if (
        buyer_code is not None
        and not valid_company_code(buyer_code)
    ):
        issues.append(
            Issue(
                field="buyer.code",
                rule="company_code",
                message=(
                    "Код покупця не проходить "
                    "перевірку контрольного розряду."
                ),
            )
        )

    # -----------------------------------------------------
    # 5. Покупець = наша компанія
    # -----------------------------------------------------

    if normalize_text(buyer.get("name")) != normalize_text(company["name"]):
        issues.append(
            Issue(
                field="buyer.name",
                rule="buyer",
                message=(
                    "Покупець не збігається з "
                    "ТОВ «Сузірʼя Рітейл»."
                ),
            )
        )

    if normalize_code(buyer.get("code")) != normalize_code(company["code"]):
        issues.append(
            Issue(
                field="buyer.code",
                rule="buyer",
                message=(
                    "Код покупця не збігається "
                    "з кодом нашої компанії."
                ),
            )
        )

    # -----------------------------------------------------
    # 6. Постачальник у довіднику
    # -----------------------------------------------------

    known_supplier = None

    current_supplier_code = normalize_code(
        supplier.get("code")
    )

    for reference_supplier in suppliers:
        if (
            normalize_code(reference_supplier.get("code"))
            == current_supplier_code
        ):
            known_supplier = reference_supplier
            break

    if known_supplier is None:
        issues.append(
            Issue(
                field="supplier.code",
                rule="supplier_reference",
                message="Постачальника немає в довіднику.",
            )
        )

    else:
        if (
            normalize_text(supplier.get("name"))
            != normalize_text(known_supplier["name"])
        ):
            issues.append(
                Issue(
                    field="supplier.name",
                    rule="supplier_reference",
                    message=(
                        "Назва постачальника не збігається "
                        "з довідником."
                    ),
                )
            )

        if (
            normalize_iban(supplier.get("iban"))
            != normalize_iban(known_supplier["iban"])
        ):
            issues.append(
                Issue(
                    field="supplier.iban",
                    rule="supplier_reference",
                    message=(
                        "IBAN постачальника не збігається "
                        "з довідником."
                    ),
                )
            )

    # -----------------------------------------------------
    # 7. Арифметика кожної позиції
    # -----------------------------------------------------

    calculated_items_total = Decimal("0")
    all_item_amounts_available = True

    for index, item in enumerate(items):
        quantity_raw = item.get("quantity")
        price = money(item.get("price"))
        amount = money(item.get("amount"))

        try:
            quantity = (
                Decimal(str(quantity_raw))
                if quantity_raw is not None
                else None
            )
        except (InvalidOperation, ValueError):
            quantity = None

        if quantity is None:
            issues.append(
                Issue(
                    field=f"items.{index}.quantity",
                    rule="required",
                    message="Кількість відсутня або некоректна.",
                )
            )

        if price is None:
            issues.append(
                Issue(
                    field=f"items.{index}.price",
                    rule="money",
                    message="Ціна відсутня або має некоректний формат.",
                )
            )

        if amount is None:
            all_item_amounts_available = False

            issues.append(
                Issue(
                    field=f"items.{index}.amount",
                    rule="money",
                    message="Сума позиції відсутня або має некоректний формат.",
                )
            )

        if quantity is not None and price is not None and amount is not None:
            expected_amount = round_money(quantity * price)

            if not money_equal(expected_amount, amount):
                issues.append(
                    Issue(
                        field=f"items.{index}.amount",
                        rule="item_arithmetic",
                        message=(
                            "Сума позиції не дорівнює "
                            "кількість × ціна."
                        ),
                    )
                )

        if amount is not None:
            calculated_items_total += amount

    # -----------------------------------------------------
    # 8. Сума позицій = subtotal
    # -----------------------------------------------------

    subtotal = money(document.get("subtotal"))

    if (
        subtotal is not None
        and all_item_amounts_available
        and items
    ):
        calculated_items_total = round_money(
            calculated_items_total
        )

        if not money_equal(
            calculated_items_total,
            subtotal,
        ):
            issues.append(
                Issue(
                    field="subtotal",
                    rule="subtotal",
                    message=(
                        "Разом без ПДВ не дорівнює "
                        "сумі всіх позицій."
                    ),
                )
            )

    # -----------------------------------------------------
    # 9. ПДВ
    # -----------------------------------------------------

    vat = money(document.get("vat"))

    if known_supplier is not None and subtotal is not None and vat is not None:
        vat_payer = bool(
            known_supplier.get("vat_payer")
        )

        if vat_payer:
            expected_vat = round_money(
                subtotal * Decimal("0.20")
            )
        else:
            expected_vat = Decimal("0.00")

        if not money_equal(expected_vat, vat):
            issues.append(
                Issue(
                    field="vat",
                    rule="vat",
                    message=(
                        f"ПДВ має бути {expected_vat} грн "
                        "відповідно до довідника постачальника."
                    ),
                )
            )

    # -----------------------------------------------------
    # 10. subtotal + VAT = total
    # -----------------------------------------------------

    total = money(document.get("total"))

    if (
        subtotal is not None
        and vat is not None
        and total is not None
    ):
        expected_total = round_money(
            subtotal + vat
        )

        if not money_equal(expected_total, total):
            issues.append(
                Issue(
                    field="total",
                    rule="total",
                    message=(
                        "Усього до сплати не дорівнює "
                        "сумі без ПДВ + ПДВ."
                    ),
                )
            )

    # -----------------------------------------------------
    # 11. Дати
    # -----------------------------------------------------

    invoice_date_raw = document.get("invoice_date")
    valid_until_raw = document.get("valid_until")

    invoice_date = parse_date(invoice_date_raw)
    valid_until = parse_date(valid_until_raw)

    if invoice_date_raw is not None:
        if invoice_date is None:
            issues.append(
                Issue(
                    field="invoice_date",
                    rule="date",
                    message="Дата рахунку некоректна.",
                )
            )
        elif invoice_date > date.today():
            issues.append(
                Issue(
                    field="invoice_date",
                    rule="date",
                    message="Дата рахунку знаходиться в майбутньому.",
                )
            )

    # valid_until не є обов'язковим.
    if valid_until_raw is not None:
        if valid_until is None:
            issues.append(
                Issue(
                    field="valid_until",
                    rule="date",
                    message="Строк дії має некоректну дату.",
                )
            )

        elif (
            invoice_date is not None
            and valid_until < invoice_date
        ):
            issues.append(
                Issue(
                    field="valid_until",
                    rule="date",
                    message=(
                        "Строк дії рахунку не може бути "
                        "раніше дати рахунку."
                    ),
                )
            )

    return issues