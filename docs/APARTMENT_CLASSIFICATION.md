# Apartment classification

Room function and shared use are separate concepts. The classifier now includes
gym, walk-in closet, building lobby and elevator lobby. A gym, pool or hallway
does not establish whether a space is private or communal.

Shot-line JSON exposes `space_scope`: currently `shared` for explicitly communal
labels and `unknown` otherwise. `private` is reserved for future verified customer
input; the automatic pipeline does not infer private ownership from appearance.
Shared is a classification prediction, not a verified ownership claim.

Shared spaces and amenities follow the home interior and secondary suite, before
the outdoor payoff. Gyms remain feature rooms; wardrobes follow service-room
exclusion. Geometry and pair thresholds are unchanged.

New labels apply to fresh classification. Existing saved labels are not rewritten.
No extra image model pass is introduced: the existing embedding classifier uses
the expanded prompt vocabulary. Before claiming improved accuracy, label the new
listing examples and compare old/new predictions on those and prior listings.
No new-listing accuracy measurement has been run yet.
