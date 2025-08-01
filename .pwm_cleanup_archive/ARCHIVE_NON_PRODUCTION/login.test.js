// login.test.js
// Automated test for sacred digital login ritual animation

import React from "react";
import { render, fireEvent, screen, act } from "@testing-library/react";
import LoginPage from "./login";

jest.useFakeTimers();

describe("Sacred Digital Login Ritual", () => {
    it("triggers particle morphing animation on login", () => {
        render(<LoginPage />);
        // Fill in username and seed
        fireEvent.change(screen.getByPlaceholderText(/alicesmith/i), {
            target: { value: "testuser" },
        });
        fireEvent.change(screen.getByPlaceholderText(/dream moon olive/i), {
            target: { value: "testseed" },
        });
        // Click authenticate
        fireEvent.click(screen.getByText(/Authenticate/i));
        // Ritual animation should be active
        expect(screen.getByRole("img", { hidden: true })).toBeDefined;
        // Fast-forward timer to end animation
        act(() => {
            jest.advanceTimersByTime(1800);
        });
        // Ritual animation should end
        // (Canvas should be cleared, status should update)
        expect(screen.getByText(/Verifying symbolic credentials/i)).toBeInTheDocument();
    });
});
