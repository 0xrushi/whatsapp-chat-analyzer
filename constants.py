pattern_attachment = [
        ".*<Media omitted>$", #English version of android attachment
        ".*image omitted$",
        ".*video omitted$",
        ".*document omitted$",
        ".*Contact card omitted$",
        ".*audio omitted$",
        ".*GIF omitted$",
        ".*sticker omitted$"
    ]
pattern_event = [
        "Messages to this group are now secured with end-to-end encryption\.$",
        ".+\screated this group$",
        ".+\sleft$",
        ".+\sadded\s.+",
        ".+\sremoved\s.+",
        ".*You joined using this group's invite links$",
        ".+'s security code changed\.$",
        ".*changed their phone number to a new number. Tap to message or add the new number\.$"
    ]
pattern_url = "https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+"
