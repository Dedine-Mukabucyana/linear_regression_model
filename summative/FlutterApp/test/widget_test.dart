import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:gbv_predictor/main.dart';

void main() {
  testWidgets('shows 9 labelled fields and a Predict button', (tester) async {
    await tester.pumpWidget(const MyApp());
    expect(find.byType(TextField), findsNWidgets(9));
    expect(find.text('Predict'), findsOneWidget);
  });
}